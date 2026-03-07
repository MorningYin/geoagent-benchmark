# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# 导入Python内置的临时文件与目录处理模块，当未指定输出目录时生成临时目录存储评测日志
import tempfile

# 导入click库，用于构建强大的命令行接口（CLI）工具
import click

# 导入ARE项目内部命令行的共享参数定义（用装饰器的方式复用命令行参数）
from are.simulation.cli.shared_params import (
    common_options,        # 通用选项（如指定模型、打印日志级别等）
    json_config_options,   # JSON 配置相关选项
    output_config_options, # 输出结果配置选项
    runtime_config_options,# 运行时配置选项（设置并发、oracle模式等）
)
# 从 utils 中导入工具函数，用于加载不同来源的Scenario、环境配置等
from are.simulation.cli.utils import (
    create_noise_configs,                 # 创建干扰/噪声配置（模拟真实环境中不可靠的因素）
    run_scenarios_by_huggingface_urls,    # 通过提供的 HuggingFace URL 在线拉取剧本并运行
    run_scenarios_by_id,                  # 通过已经在代码中注册好的字符串 ID 来运行剧本
    run_scenarios_by_json_files,          # 通过本地指定的 JSON 文件路径来运行剧本
    setup_logging,                        # 设置统一个的系统日志格式和打印级别
)
# 导入配置数据结构类，用于将命令行散装的参数打包成一个聚合配置对象传递给后续线程池和环境
from are.simulation.scenarios.config import MultiScenarioRunnerConfig
# 导入 HuggingFace URL 解析器，将带 hf:// 的地址拆解为具体的 dataset, config, split, scenario_id
from are.simulation.utils.huggingface import parse_huggingface_url


def validate_main_scenario_sources(**scenario_params):
    """
    【校验 CLI 获取的剧本数据源是否合法】
    
    这个函数负责验证传进来的剧本来源是否合规。它会处理三种剧本来源：
    1. scenario_id（内置剧本名）
    2. json_file / scenario_file（本地剧本 JSON）
    3. hf_url（HuggingFace URL 在线剧本）
    
    要求：这三种来源互相排斥，你一次只能指定其中一种，如果同时提供了多种或者一种都没提供，就抛出命令行使用异常。

    :param scenario_params: 包含各种剧本来源参数的字典
    :type scenario_params: dict
    :returns: 返回一个元组包含激活的 (scenario_ids, scenario_files, hf_urls)
    :rtype: tuple[list[str] | None, list[str] | None, list[str] | None]
    :raises click.UsageError: 当校验失败时抛出 Click 的标准用法异常（直接将错误打到控制台并退出）
    """
    # ====== 第一步：从参数字典中提取用户命令行传进来的各种剧本来源 ======
    scenario_id = scenario_params.get("scenario_id")
    scenario_file = scenario_params.get("scenario_file")
    hf_url = scenario_params.get("hf_url")

    # 为了兼容旧版本，还提取老的 json_file 参数
    json_file = scenario_params.get("json_file")

    # ====== 第二步：合并新旧参数 ======
    final_scenario_id = scenario_id
    final_scenario_file = json_file

    # 如果有新的 scenario_file 且没有老的 json_file，就用新的
    if scenario_file and not json_file:
        final_scenario_file = scenario_file
    # 如果用户既传了旧的 json_file 又传了新的 scenario_file，系统不知道听谁的，直接报错
    elif scenario_file and json_file:
        raise click.UsageError(
            "Cannot specify both --scenario-file and --json_file. Use --scenario-file instead."
        )

    # ====== 第三步：校验 HuggingFace Url 的格式合法性 ======
    if hf_url:
        for url in hf_url:
            hf_params = parse_huggingface_url(url) # 通过正则或者切片解析 url 获取各级名称
            # 如果解析失败（返回 None），说明 URL 格式不符合规范
            if hf_params is None:
                raise click.UsageError(
                    f"Invalid HuggingFace URL format: '{url}'. Expected format: hf://datasets/dataset_name/config/split/scenario_id"
                )

    # ====== 第四步：清点启用了几种剧本来源源头 ======
    # 把存在内容的列表/字符串判定为 True(1)，然后累加。正常情况下结果只能为 1
    sources_specified = sum(
        [bool(final_scenario_id), bool(final_scenario_file), bool(hf_url)]
    )

    # 如果一个是都没填，报错
    if sources_specified == 0:
        raise click.UsageError(
            "Must specify one of: --scenario-id/--scenario_id, --scenario-file/--json_file, or --hf-url"
        )
    # 如果填了多种（比如又查 ID 又查 URL），报错。系统一次执行只能用一种加载协议
    elif sources_specified > 1:
        raise click.UsageError(
            "Cannot specify multiple scenario sources at the same time. Choose one of: --scenario-id, --scenario-file, or --hf-url"
        )

    # 校验全部通过，返回梳理归一化后的三个变量
    return final_scenario_id, final_scenario_file, hf_url


# @click.command() 将这个 main 函数注册成命令行的主命令
@click.command()
# ====== 挂载各种公用命令行参数（使用复用的装饰器） ======
@common_options()          # 提供如 --model（要给用什么基础模型跑）这些基础参数
@runtime_config_options()  # 提供运行时需要的多线程并发、oracle测试模式等参数
@json_config_options()     # 获取与 json 加载及参数覆盖相关的配置项
# ====== 针对这个入口特定的参数定义开始 ======
@click.option(
    "-s",
    "--scenario-id",
    required=False,     # 可选参数
    multiple=True,      # 这个参数在启动时能写多次（比如 -s Test1 -s Test2）
    help="从已注册库中运行指定的 Scenario ID（可以指定多次以运行多个剧本）",
)
@click.option(
    "--scenario-file",
    type=click.Path(exists=True, dir_okay=False, path_type=str), # 要求必须是存在的文件路径，不能是文件夹
    required=False,
    multiple=True,
    help="运行指定的 JSON 格式剧本文件（可以指定多次）",
)
@click.option(
    "--hf-url",
    required=False,
    multiple=True,
    help="运行指定 HuggingFace 数据集上的剧本，格式必须是: hf://datasets/dataset_name/config/split/scenario_id",
)
# ====== 向下兼容的废弃参数 ======
@click.option(
    "--scenario_id",
    required=False,
    multiple=True,
    help="Scenarios to run (deprecated, use --scenario-id instead)",
    hidden=True, # 不在命令行 --help 文档中显示
)
@click.option(
    "-j",
    "--json_file",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    required=False,
    multiple=True,
    help="JSON scenario files to run (deprecated, use --scenario-file instead)",
    hidden=True,
)
@output_config_options()  # 挂载输出结果路径等配置
@click.option(
    "-e",
    "--export",
    is_flag=True,   # 布尔值标志位，写了 -e 就是 True
    default=False,
    help="是否将执行全过程的 Trace 轨迹导出为 JSON 文件。",
)
@click.option(
    "-w",
    "--wait-for-user-input-timeout",
    type=float,
    default=None,
    help="给等待用户输入设定的超时时间/秒。如果不设置则永久等待（这在人机交互下才需要）。",
)
@click.option(
    "--list-scenarios",
    is_flag=True,
    default=False,
    help="只列出所有已经注册在库里的剧本名称，然后立刻退出。",
)
# ====== 命令行函数的参数体，参数名与上面装饰器生命的选项对应 ======
def main(
    model: str,                                    # 智能体的底座模型 (如 "gpt-4-turbo")
    provider: str | None = None,                   # 模型供应商名 (如 "openai")，非必填
    endpoint: str | None = None,                   # API自建端点
    agent: str | None = None,                      # 要使用的特定 Agent 类名 / 配置名，不填代表无人机挂机空转
    log_level: str = "INFO",                       # 控制台打印的日志详细程度 ("DEBUG", "INFO", "WARNING")
    oracle: bool = False,                          # 上帝模式：给基准测试做答案标定时使用的模式，忽略大模型操作预测
    simulated_generation_time_mode: str = "measured", # 如何计算模型生成时间（是实际流逝时间还是根据输出 token 数预估）
    noise: bool = False,                           # 是否开启真实环境中的噪音干扰（如网络延迟、工具随机失效等测试）
    max_concurrent_scenarios: int | None = None,   # 限制线程池多线程并行的数量
    kwargs: str = "{}",                            # 通用的通过字符串传递给剧本初始化时的额外 JSON 参数，每个都有
    multi_kwargs: str = "[{}]",                    # 数组形式传递的各个剧本独有的额外初始化 json 参数
    scenario_kwargs: str = "{}",                   # 通用的创建 Scenario 时传入的额外参数
    multi_scenario_kwargs: str = "[{}]",           # 数组形式给多个 Scenario 分别传入的不同开局配置参数
    # 以下三个由上方的专门选项传入
    scenario_id: list[str] | None = None,          
    scenario_file: list[str] | None = None,
    hf_url: list[str] | None = None,
    # 旧版本参数
    json_file: list[str] | None = None,            
    output_dir: str | None = None,                 # 日志或评测轨迹回放导出存放的文件夹绝对路径
    export: bool = False,                          # 控制是否执行完整的 Trace 运行轨迹导出操作
    wait_for_user_input_timeout: float | None = None, # 模拟器对于人类手动输入的挂起忍受阈值
    list_scenarios: bool = False,                  # 控制是否直接打印存在的注册剧本列表并且立即跳出应用
):
    """
    【ARE 主程序入口点】
    
    这是 `are-run` 命令行脚本真实执行的核心逻辑。
    它负责整理接收到的所有命令行指令参数，构造 MultiScenarioRunnerConfig 统一配置类，
    并将指定的剧本(Scenarios)来源按不同渠道加载，并抛给底层执行器执行。
    """
    # 按照所选级别（INFO/DEBUG）以及预设固定格式配置全局的标准 Python 日志分类及时间记录等基础库
    setup_logging(log_level)

    # 【分支1】: 如果用户带了 --list-scenarios 标志位
    if list_scenarios:
        # 懒加载（走到这里才import）获取当前环境内全局存在的 registry（用来存所有使用修饰器注册或者预加载的静态类的全局字典缓存）
        from are.simulation.scenarios.utils.registry import registry

        scenarios = registry.get_all_scenarios() # 读取当前注册表内的所有剧本集合副本
        if not scenarios:
            click.echo("No scenarios are currently registered.")
        else:
            # 取出字典所有的 key 然后字典序排序打印出来供用户在控制台上检阅
            for scenario_key in sorted(scenarios.keys()):
                click.echo(scenario_key)
        # 打印完毕立马主动终止结束整个程序的后续操作
        return

    # 【分支2】：调用上方自己写的内部校验函数，来互斥排查并且正规化梳理用户指定了究竟选择哪种剧本的数据集加载来源途径
    final_scenario_id, final_json_file, final_hf_url = validate_main_scenario_sources(
        scenario_id=scenario_id,
        scenario_file=scenario_file,
        hf_url=hf_url,
        json_file=json_file,
    )

    # 声明干扰项配置文件，准备开启噪音测试模式时再深度生成它们
    tool_augmentation_config = None
    env_events_config = None
    if noise:
        # 该函数会生成并填充诸如“工具产生随机调用失败概率分布表”、“随机环境突然宕机概率”等配置
        tool_augmentation_config, env_events_config = create_noise_configs()

    # 安全后备检查项：如果用户需要导出结果或者要存运行的 trace log，但又没有指定具体放在本机的什么目录下
    if output_dir is None:
        # 利用系统自带的标准库 `tempfile` 在系统 /tmp 下开辟一个唯一的不重叠的前缀含有 "are_simulation_output_" 的专用临时匿名文件夹目录
        output_dir = tempfile.mkdtemp(prefix="are_simulation_output_")

    # ====== 核心总线整合：把刚刚分散定义的这么多变量全部打包，构建制造成一个不可修改生命周期的底层运行总基线不可变配置对象 ======
    # 下游接管的整个并发底座多线程执行引擎 (MultiScenarioRunner) 和每一个游玩具体世界 (ScenarioRunner) 只会接纳传递这单一封装好的一个配置对象。
    runner_config = MultiScenarioRunnerConfig(
        model=model,
        model_provider=provider,
        agent=agent,
        scenario_creation_params=scenario_kwargs,
        scenario_multi_creation_params=multi_scenario_kwargs,
        scenario_initialization_params=kwargs,
        scenario_multi_initialization_params=multi_kwargs,
        oracle=oracle,
        export=export,
        wait_for_user_input_timeout=wait_for_user_input_timeout,
        output_dir=output_dir,
        endpoint=endpoint,
        max_concurrent_scenarios=max_concurrent_scenarios,
        simulated_generation_time_mode=simulated_generation_time_mode,
        tool_augmentation_config=tool_augmentation_config,
        env_events_config=env_events_config,
        enable_caching=False, # 默认在基准评测批量跑分模式运行(即当前使用 CLI入口)时是彻底强制关闭任何短路换取的重跑结果依赖的
    )

    # ====== 分发路由跳转阶段：把封装好的超大配置参数结构，连同确切可执行的剧本来源资源坐标，移交给底层执行具体加载解析并循环流转数据的挂起函数执行 ======
    
    # 途径一：剧本是通过字符串内置硬编码好的类名 ID 强制被明确指定的，在本地运行，直接去找全局本地内存静态 registry 池子读取匹配的子类并执行启动
    if final_scenario_id:
        run_scenarios_by_id(runner_config, final_scenario_id)
        return

    # 途径二：传入的变量数组是磁盘上的一个或几个特定存在、验证过的绝对相对路径 json 文件群，通过底层的 JsonDeserializer 解析还原并模拟实例化该类运行
    if final_json_file:
        run_scenarios_by_json_files(runner_config, final_json_file)
        return

    # 途径三：是在线地址，向网络发出 HTTP/HF Api 查询指令包，拉取网络临时流、解析格式，构建虚拟临时实例，并装载启动跑动进程。
    if final_hf_url:
        run_scenarios_by_huggingface_urls(runner_config, final_hf_url)
        return


if __name__ == "__main__":
    # 当自身被 Python 解释器当作主运行脚本 (即 python main.py) 执行而非只是包导入时启动整个控制流树
    main()

