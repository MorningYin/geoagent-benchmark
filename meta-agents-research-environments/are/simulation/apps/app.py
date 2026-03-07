# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# 导入Python内置库
import inspect # 用于获取类和方法的内部信息，比如参数、文档字符串等
import logging # 用于记录系统的运行日志
import random # 用于生成随机数，这里主要用于模拟工具调用的随机失败
from abc import ABC # 引入抽象基类模块，用于定义App必须实现的接口
from enum import Enum, auto # 用于定义枚举类型
from typing import Any, Callable # 用于类型提示，Callable代表函数或方法，Any代表任意类型

# 导入ARE项目内部的工具和工具类
from are.simulation.time_manager import TimeManager # 虚拟时钟管理器
from are.simulation.tool_utils import AppTool, ToolAttributeName, build_tool # 工具定义和解析的核心类
from are.simulation.utils import SkippableDeepCopy, add_reset # 拷贝和状态重置的辅助工具

logger = logging.getLogger(__name__) # 初始化日志记录器


class Protocol(Enum):
    """
    定义应用之间通信的协议。
    当你希望一个App调用另一个App的底层能力时（比如发邮件需要读文件），可以通过协议挂载。
    目前实现了文件系统协议。
    """
    FILE_SYSTEM = "FILE_SYSTEM"


class ToolType(Enum):
    """
    定义工具的不同类型（或者说是给不同角色用的工具）。
    通过这种分类，可以限制谁能使用哪些工具。
    """
    APP = auto()  # 供AI Agent（智能体）调用的工具
    USER = auto() # 供模拟人类用户调用的工具
    ENV = auto()  # 供环境或场景设计者使用的初始化工具
    DATA = auto() # 供数据加载工具使用的数据填充工具


@add_reset # 这是一个装饰器，给类添加自动重置内部状态的能力
class App(ABC, SkippableDeepCopy):
    """
    Meta Agents Research Environments 虚拟环境里所有"应用程序"的抽象基类。
    不管你是邮箱(EmailClient)、日历(Calendar)还是本地文件系统，都必须继承这个类。
    """

    # 由于在App被注册到环境后，它会持有对environment(环境实例)的引用
    # 而environment里又有一些无法被序列化的东西（比如线程锁 threading.lock）
    # 所以我们在做"深拷贝(deepcopy)"或"序列化(pickling)"存盘时，需要跳过这些字段
    _skip_deepcopy_fields = ["add_event", "add_event_callbacks"]
    _skip_pickle_fields = _skip_deepcopy_fields

    def __init__(self, name: str | None = None, *args, **kwargs):
        super().__init__()
        # 设定App的名字，如果没有传入指定名字，就用类名
        self.name = self.__class__.__name__ if name is None else name
        self.is_state_modified = False # 记录状态是否被修改，可用于判断是否需要存档
        
        # 存储环境注入进来的回调函数。通过调用这些回调，App可以将新事件塞进Environment的事件队列里
        self.add_event_callbacks = {}
        
        # 工具注册表缓存：用于分类存放解析出来的四种工具，避免每次都要通过反射去扫描类方法
        self._tool_registries: dict[ToolType, list[AppTool] | None] = {
            ToolType.APP: None,
            ToolType.USER: None,
            ToolType.ENV: None,
            ToolType.DATA: None,
        }
        
        # 注入故障概率：为了测试AI遭遇失败时的恢复能力，我们可以让Agent调用的工具按一定概率(比如0.1)随机返回失败
        self.failure_probability: float | None = None
        self.time_manager = TimeManager() # 每个App自带时间管理器参考
        self.set_seed(0) # 初始化随机数种子

    def register_time_manager(self, time_manager: TimeManager):
        """由Environment在初始化时调用，确保所有App使用同一个环境虚拟时间"""
        self.time_manager = time_manager

    def set_seed(self, seed: int) -> None:
        """
        设置随机种子。
        通过组合传入的seed和App的名字，推导出一个新的专属seed。
        这确保了即使使用了相同的初始seed，每个不同的App实例也会有独立但确定性的随机序列。
        """
        combined_seed = f"{seed}_{self.name}"
        self.seed = hash(combined_seed) % (2**32)
        self.rng = random.Random(self.seed)

    def register_to_env(
        self,
        key: str,
        add_event: Callable[[Any], None],
    ):
        """
        将App注册到环境（Environment）时的回调注入。
        :param key: 回调的标识符
        :param add_event: environment提供的函数，用于将新的Event添加到主循环中
        """
        self.add_event_callbacks[key] = add_event

    def add_event(self, event: Any) -> None:
        """
        应用程序主动向虚拟环境触发一个事件。
        比如：收到了一封邮件导致了某个反应流程。
        """
        for callback in self.add_event_callbacks.values():
            callback(event)

    def get_implemented_protocols(self) -> list[Protocol]:
        """
        返回当前App具体实现了哪些协议服务（比如返回[Protocol.FILE_SYSTEM]）。
        其他App可以通过这个列表知道是否能依赖本App的服务。
        """
        return []

    def connect_to_protocols(self, protocols: dict[Protocol, Any]) -> None:
        """
        接受并挂载其他App提供的协议服务。
        如果当前App需要读写文件，Environment就会把实现了FileSystem协议的App通过这里传进来。
        """
        pass

    def get_state(self) -> dict[str, Any] | None:
        """导出App当前在内存中的所有状态（数据字典），用于存档/重置。"""
        pass

    def load_state(self, state_dict: dict[str, Any]):
        """从之前导出的字典中恢复App的状态。"""
        pass

    def reset(self):
        """重置App的流转状态，比如恢复随机数生成器的初始状态。通过装饰器 @add_reset 还可以级联清理其它数据。"""
        self.rng = random.Random(self.seed)

    def app_name(self) -> str:
        """获取当前App的名称。"""
        return self.name

    def set_failure_probability(self, failure_probability: float) -> None:
        """
        设置故障概率。这是Arena测试AI韧性的机制。
        设置完概率后，必须要清空注册表缓存，因为构建工具对象时需要把这个概率注进去。
        """
        logger.debug(f"Setting failure_probability to {failure_probability}")
        self.failure_probability = failure_probability
        # 由于故障概率改变了，我们需要让工具重新生成一次
        logger.debug("Resetting tool registries")
        self._tool_registries: dict[ToolType, list[AppTool] | None] = {
            ToolType.APP: None,
            ToolType.USER: None,
            ToolType.ENV: None,
            ToolType.DATA: None,
        }

    def get_tools_with_attribute(
        self, attribute: ToolAttributeName, tool_type: ToolType
    ) -> list[AppTool]:
        """
        （核心功能）遍历当前类和所有父类的所有方法，找出带有特定装饰器标记的方法，并将它们转换成给LLM调用的工具格式(AppTool)。

        :param attribute: 工具特性的枚举值（检查方法上是否存在特定的属性标记，比如 '_is_app_tool'）
        :type attribute: ToolAttributeName
        :param tool_type: 正在注册的工具类型类别 (APP/USER/ENV/DATA)
        :type tool_type: ToolType
        :return: 解析包装后的AppTool对象列表
        :rtype: list[AppTool]
        """
        # 获取枚举对应的实际字符串值（如 '_is_app_tool'）
        attr_name = attribute.value
        tools = []
        processed_attributes = set()  # 记录已处理过的方法名，防止父类同名方法被重复注册
        cls = self.__class__

        # 遍历类及其所有父类 (Method Resolution Order, MRO)
        # 子类会被优先遍历。这能保证如果子类重写(override)了父类方法，我们只会把子类版本注册为工具
        for base_cls in inspect.getmro(cls):
            for attr_name, attr_value in base_cls.__dict__.items():
                if attr_name in processed_attributes:
                    # 如果子类已经注册过这个名字的函数了，直接跳过父类的
                    continue

                # 兼容性处理：如果传进来的是枚举，就取值；如果是字符串，直接用
                attr_name_str = (
                    attribute.value
                    if isinstance(attribute, ToolAttributeName)
                    else attribute
                )
                
                # 判断当前方法(attr_value)身上有没有通过装饰器打上的标记（比如 hasattr(func, '_is_app_tool')）
                if hasattr(attr_value, attr_name_str):
                    logger.debug(
                        f"[Registering {tool_type} Tool] {attr_name} of class {base_cls.__name__}"
                    )
                    # 给LLM用的工具必须要有docstring才能起作用，没有的话尝试去这行代码父类里借一个
                    if not attr_value.__doc__:
                        logger.error(
                            f"\tDid not find doc of {attr_name} of class {base_cls.__name__} - trying base class method"
                        )
                        attr_value.__doc__ = get_base_method_doc(base_cls, attr_value)

                    # 仅仅对于供AI使用的 APP 工具，我们才应用随机故障模拟；诸如环境预设(ENV)是系统行为，就不让它随机失败了
                    failure_probability = (
                        self.failure_probability if tool_type == ToolType.APP else None
                    )
                    
                    # build_tool 方法会解析函数的参数、类型和文档，整合成完整的 JSON Schema / OpenAI function 对象
                    tools.append(build_tool(self, attr_value, failure_probability))
                    processed_attributes.add(attr_name)  # 标记已处理

        # 另外通过遍历实例自身的 __dict__ 检查有没有动态挂载到实例上的方法被标记为工具
        for attr_name, attr_value in self.__dict__.items():
            if attr_name in processed_attributes:
                continue

            attr_name_str = (
                attribute.value
                if isinstance(attribute, ToolAttributeName)
                else attribute
            )
            if hasattr(attr_value, attr_name_str):
                logger.debug(
                    f"[Registering {tool_type} Tool] {attr_name} of instance {self.__class__.__name__}"
                )
                if not attr_value.__doc__:
                    logger.error(
                        f"\tDid not find doc of {attr_name} of instance {self.__class__.__name__}"
                    )

                failure_probability = (
                    self.failure_probability if tool_type == ToolType.APP else None
                )
                tools.append(build_tool(self, attr_value, failure_probability))
                processed_attributes.add(attr_name)

        logger.debug(
            f"[Registering {tool_type} Tool] Built Tool Registry for class {cls.__name__} with {len(tools)} tools"
        )

        return tools

    def _get_or_initialize_tools(
        self, tool_type: ToolType, attribute: ToolAttributeName
    ) -> list[AppTool]:
        """
        工具获取器的内部缓存机制助手。

        :param tool_type: 工具按角色分类 (例如 APP / USER / ENV)
        :param attribute: 传给扫描器的实际判断属性名称
        :return: AppTool列表
        """
        # 如果还没解析过（缓存为空），就调用上面的长方法去反射扫描，扫描完存进缓存
        if self._tool_registries[tool_type] is None:
            tools = self.get_tools_with_attribute(
                attribute=attribute, tool_type=tool_type
            )
            self._tool_registries[tool_type] = tools
            return tools
        # 有缓存就直接返回
        return self._tool_registries[tool_type] or []

    def get_tools(self) -> list[AppTool]:
        """
        获取属于Agent(AI)使用的普通应用工具列表（被 @app_tool 标记的方法）。
        """
        return self._get_or_initialize_tools(ToolType.APP, ToolAttributeName.APP)

    def get_user_tools(self) -> list[AppTool]:
        """
        获取属于人类或模拟用户的操作工具列表（被 @user_tool 标记的方法）。
        """
        return self._get_or_initialize_tools(ToolType.USER, ToolAttributeName.USER)

    def get_env_tools(self) -> list[AppTool]:
        """
        获取属于场景设计时初始化环境状态用的环境工具列表（被 @env_tool 标记的方法）。
        """
        return self._get_or_initialize_tools(ToolType.ENV, ToolAttributeName.ENV)

    def get_tool(self, tool_name: str) -> AppTool | None:
        """
        如果知道工具的具体方法名，无论它是哪种角色类型，通过全局搜查找出它的封装对象。
        :param tool_name: 所寻找的工具短名称 (比如: send_email)
        :rtype: AppTool | None
        """
        # 依次在四类工具清单中搜索
        for tool_getter in [
            self.get_tools,
            self.get_user_tools,
            self.get_env_tools,
            self.get_data_tools,
        ]:
            try:
                tools = tool_getter()

                # 寻找匹配名字的工具。构建Tool时，命名格式是 '{应用名称}__{方法名称}'
                for tool in tools:
                    if tool.name == f"{self.name}__{tool_name}":
                        return tool
            except Exception:
                # 即使某一类在获取时出错，也继续查找下一类
                continue
            return None

    def get_data_tools(self) -> list[AppTool]:
        """
        获取用于数据拉取或处理的数据工具列表（被 @data_tool 标记的方法）。
        """
        return self._get_or_initialize_tools(ToolType.DATA, ToolAttributeName.DATA)

    def pause_env(self) -> None:
        """用于主动暂停虚拟环境的空接口方法，可供子类实现特定的暂停逻辑"""
        pass

    def resume_env(self) -> None:
        """用于恢复虚拟环境的空接口方法，可供子类实现特定的恢复逻辑"""
        pass


def get_base_method_doc(cls: type, method: Callable) -> str | None:
    """
    备用：如果子类的方法忘记写docstring了，它可以在这里顺延找父类同名方法的docstring。

    :param cls: 当前遍历到的类
    :type cls: type
    :param method: 需要获取文档字符串的方法引用
    :type method: Callable

    :return: 找到的说明文档，如果没有就返回 None
    """
    # 遍历当前类的直接父类
    for base in cls.__bases__:
        base_method = getattr(base, method.__name__, None)
        # 如果父类有这个方法且配有详细的文档注释，就直接借过来
        if base_method and base_method.__doc__:
            return base_method.__doc__
    return None
