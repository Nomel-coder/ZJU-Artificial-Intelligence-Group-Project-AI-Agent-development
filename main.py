import sys
from agent import ReActAgent

def main():
    print("=" * 60)
    print(" ReAct AI Agent CLI - 终端交互版")
    print("=" * 60)
    print("输入你的问题（例如：“计算圆周率 3.14159 乘以半径 10 的平方，并将结果写入 area.txt”）")
    print("输入 'exit' 或 'quit' 退出。")
    print("-" * 60)
    
    try:
        agent = ReActAgent()
    except Exception as e:
        print(f"初始化 Agent 失败: {str(e)}")
        sys.exit(1)
        
    while True:
        try:
            user_input = input("\n用户 > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("退出程序。")
                break
                
            print("\n开始执行 ReAct 推理闭环...")
            steps_generator = agent.run(user_input)
            
            for step in steps_generator:
                step_type = step["type"]
                if step_type == "thought":
                    print(f"\n\033[36m[Thought 思考]\033[0m\n{step['content']}")
                elif step_type == "tool_call":
                    print(f"\n\033[33m[Action 行为]\033[0m 调用工具: \033[1m{step['tool']}\033[0m")
                    print(f"  └─ 参数: {step['input']}")
                elif step_type == "observation":
                    print(f"\n\033[32m[Observation 观察]\033[0m 工具返回结果:\n{step['content']}")
                elif step_type == "final_answer":
                    print(f"\n\033[35m\033[1m[Final Answer 最终回答]\033[0m\n{step['content']}")
                elif step_type == "error":
                    print(f"\n\033[31m[System Error 系统错误]\033[0m {step['content']}")
                    
            # 重置会话历史以开始下一个独立任务
            agent.reset()
            
        except KeyboardInterrupt:
            print("\n退出程序。")
            break
        except Exception as e:
            print(f"发生未知错误: {str(e)}")

if __name__ == "__main__":
    main()
