# OpenArm Teleop Sim

第一阶段：不用实体硬件，用 MuJoCo 里的 OpenArm v1 双臂（立柱款，16 个执行器：每臂 7 关节 + 夹爪），打通"控制指令 → 边缘安全层 → 状态反馈 → 浏览器前端"。

## 运行

```bash
./scripts/fetch_model.sh                 # 拉取 OpenArm MuJoCo 模型（固定 commit）
uv run pytest                            # 安全层 + 仿真适配器测试
uv run python -m edge.server             # http://127.0.0.1:8765
```

操作顺序：Claim control → Activate → 拖动滑块（按 Left arm / Right arm 分组）。单臂模型：`--model vendor/openarm_mujoco/v1/openarm.xml`。空格键 = E-stop。勾选 "Stop sending commands" 可模拟断网。

## 结构（与真实边缘设备一一对应）

| 文件 | 作用 | 换成真机时 |
|---|---|---|
| `edge/mujoco_adapter.py` | 机器人替身：关节名、限位、读状态、写目标、相机画面 | 换成 ROS 2 或供应商 SDK 适配器，接口不变 |
| `edge/safety.py` | 本地安全层：限位裁剪、速度限制、0.5 s 看门狗、E-stop 锁存 | 原样保留，跑在机器人本地 |
| `edge/server.py` | 100 Hz 本地控制循环 + WebSocket；单操作员控制权；JSONL 会话日志（`logs/`） | 保留；公网暴露前必须加认证和 TLS |
| `web/` | 操作员界面：视频、滑块、模式、RTT | 保留 |

协议：浏览器以 20 Hz 发送 `cmd`（目标位姿），边缘端以 20 Hz 回 `state`，约 14 fps 回 JPEG。高频伺服不经过网络。

## 已验证

- 目标超出限位会被裁剪；setpoint 移动速度不超过 1 rad/s。
- 停发指令 0.5 s 后进入 HOLD，停在当前 setpoint；重新 Activate 时不跳变。
- 反复 HOLD/Activate 不会因重力下垂而累积漂移。
- E-stop 锁存，必须重新 Activate；任何连接的客户端都能触发 E-stop。

## 这一阶段证明不了的

- 真实网络下的延迟和抖动（本地 RTT 约 1 ms）。
- 供应商机器人的真实接口、固件版本和控制器行为。
- 底盘、拖管等浇水任务本身的物理难点（这里只有固定立柱上的双臂）。
- 视频走 WebSocket JPEG，只适合仿真；跨洋链路需要换成 WebRTC。

运行原理图解（中文）：[docs/运行原理.html](docs/运行原理.html)
