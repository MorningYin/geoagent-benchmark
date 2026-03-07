# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import logging
import sys
import threading

from tqdm import tqdm

# Thread-local storage for scenario IDs
_thread_local = threading.local()


def set_logger_scenario_id(scenario_id: str, run_number: int | None = None) -> None:
    """
    Set the scenario ID and optional run number for the current logger context.

    Args:
        scenario_id: The scenario ID to associate with the current logger
        run_number: The run number to associate with the current logger (optional)
    """
    _thread_local.scenario_id = scenario_id
    _thread_local.run_number = run_number


def get_logger_scenario_id() -> str | None:
    """
    Get the scenario ID for the current logger context.

    Returns:
        The scenario ID associated with the current logger, or None if not set
    """
    return getattr(_thread_local, "scenario_id", None)


def get_logger_run_number() -> int | None:
    """
    Get the run number for the current logger context.

    Returns:
        The run number associated with the current logger, or None if not set
    """
    return getattr(_thread_local, "run_number", None)


class TqdmLoggingHandler(logging.Handler):
    """
    A custom logging handler that writes log messages using tqdm.write().
    This ensures that log messages don't interfere with tqdm progress bars.
    """

    def emit(self, record):
        try:
            msg = self.format(record)
            # Use tqdm.write with a file parameter to ensure consistent output
            # This helps avoid conflicts in multiprocess environments
            tqdm.write(msg, file=sys.stdout)
            self.flush()
        except (BrokenPipeError, OSError):
            # Handle broken pipe errors that can occur in multiprocess environments
            # when the main process terminates while workers are still writing
            pass
        except Exception:
            self.handleError(record)


class ScenarioAwareFormatter(logging.Formatter):
    """
    A custom formatter that includes the scenario ID in log messages if available
    and supports colored output based on log level.
    """

    # Color codes
    grey = "\x1b[38;20m"
    bold_yellow = "\x1b[33;1m"
    red = "\x1b[31;20m"
    green = "\x1b[32;20m"
    bold_red = "\x1b[31;1m"
    bold_white = "\x1b[37;1m"
    reset = "\x1b[0m"

    # Base format
    base_format = (
        "%(asctime)s - %(threadName)s - %(levelname)s - %(name)s - %(message)s"
    )

    # Color formats for different log levels
    FORMATS = {
        logging.DEBUG: grey + base_format + reset,
        logging.INFO: base_format,
        logging.WARNING: bold_yellow + base_format + reset,
        31: reset + base_format + reset,  # Custom level
        32: green + base_format + reset,  # Custom level
        33: bold_white + base_format + reset,  # Custom level
        logging.ERROR: red + base_format + reset,
        logging.CRITICAL: bold_red + base_format + reset,
    }

    def format(self, record):
        # Get the scenario ID and run number for the current thread
        scenario_id = get_logger_scenario_id()
        run_number = get_logger_run_number()

        # Add scenario ID and run number to the record if available
        if scenario_id and not hasattr(record, "scenario_id"):
            record.scenario_id = scenario_id

            # Build the prefix with scenario ID and run number if available
            if run_number is not None:
                prefix = f"[Scenario = {scenario_id}, Run = {run_number}]"
            else:
                prefix = f"[Scenario = {scenario_id}]"

            # Only add the prefix if the message doesn't already have it
            # Convert message to string first to handle non-string types (e.g., exceptions)
            msg_str = str(record.msg)
            if not msg_str.startswith(prefix):
                record.msg = f"{prefix} {msg_str}"

        # Apply color formatting based on log level
        log_fmt = self.FORMATS.get(record.levelno, self.base_format)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def configure_logging(level: int = logging.INFO, use_tqdm: bool = False) -> None:
    """
    【ARE 日志系统的核心组装函数】

    这里用"搭积木"的方式把 Python logging 系统的三个核心零件组装起来：
      - Formatter（日志长什么样）
      - Handler （日志写到哪儿去）
      - Logger  （谁负责收集这条日志）

    Args:
        level: 整数形式的日志级别（如 logging.INFO = 20）
        use_tqdm: 是否兼容 tqdm 进度条模式
    """

    # ── 第一步：创建"格式模板" ──────────────────────────────────────────────────
    # ScenarioAwareFormatter 是自定义类，它在父类 Formatter 基础上增加了：
    #   1. 自动从当前线程的 threading.local 里读取 scenario_id，前缀拼入日志消息
    #   2. 按日志级别（DEBUG/INFO/WARNING/ERROR）用 ANSI 转义字符给文字染色
    standard_formatter = ScenarioAwareFormatter()

    # ── 第二步：创建"输出目的地（Handler）" ─────────────────────────────────────
    # Handler 决定了日志消息最终被写到哪里（终端 / 文件 / 网络 等）
    if use_tqdm:
        # 批量评测时通常开着 tqdm 进度条。普通 print/StreamHandler 直接写 stdout
        # 会把进度条"打断"，导致输出错乱。
        # TqdmLoggingHandler 改用 tqdm.write() 输出，
        # tqdm.write 会在不破坏进度条显示的前提下插入一行文字。
        console_handler = TqdmLoggingHandler()
    else:
        # 普通运行时直接把日志输出到标准输出（终端）
        console_handler = logging.StreamHandler(stream=sys.stdout)

    # 把 Handler 自己的过滤级别也设置好（低于这个级别的消息直接在 Handler 这层被丢弃）
    console_handler.setLevel(level)
    # 把上面做好的"格式模板"挂到这个 Handler 上，让它知道每条消息要怎么排版打印
    console_handler.setFormatter(standard_formatter)

    # ── 第三步：配置"根 Logger（Root Logger）" ───────────────────────────────────
    # Python logging 里所有 Logger 构成一棵树，根节点叫 root logger。
    # 任何没有特别配置的 logger，消息最终都会"冒泡（propagate）"到这里来。
    root_logger = logging.getLogger()   # 不传名字就拿到 root logger
    root_logger.setLevel(level)         # 设置这棵树的最低接收门槛

    # 防止重复注册：每次调用这个函数都先把旧的 handler 全部清空，
    # 否则同一条日志会被打印两遍（每个 handler 各打一次）。
    # 注意用 handlers[:] 做浅拷贝再遍历，因为不能在迭代原列表时修改它。
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 把我们组装好的 console_handler 挂上去
    root_logger.addHandler(console_handler)

    # propagate=True 意思是：root logger 自己也往上传（但它已经是根了，不影响）
    # 这行写了等于没写，只是显式标明意图
    root_logger.propagate = True

    # ── 第四步：单独配置 "simulation" 命名空间 Logger ────────────────────────────
    # ARE 项目内部所有文件都用 logging.getLogger(__name__) 创建 logger，
    # __name__ 就是模块路径，比如 "are.simulation.apps.email_client"。
    # 名字以 "simulation" 开头的这些模块，会自动归属到这个 子树 Logger 下面。
    are_simulation_logger = logging.getLogger("simulation")
    are_simulation_logger.setLevel(level)

    # 同样先清空旧 handler
    for handler in are_simulation_logger.handlers[:]:
        are_simulation_logger.removeHandler(handler)

    # 挂上同一个 console_handler（和 root logger 共用同一个 handler 实例）
    are_simulation_logger.addHandler(console_handler)

    # ★ 关键一行：propagate = False
    # 正常情况下，子 Logger 处理完日志后，还会把消息向上传给父 Logger（root logger），
    # 然后 root logger 再打一遍 —— 导致同一行日志在终端出现两次。
    # 设为 False 后，"simulation.*" 的日志在这个 Logger 处理完就截住，不再向上冒泡。
    are_simulation_logger.propagate = False
