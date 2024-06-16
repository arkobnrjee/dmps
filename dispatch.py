import subprocess
import time


commands = []

for seed in range(5):
    for env in ["ObstaclesCar", "ObstaclesPoint", "GateCar", "GatePoint", "DoubleGateCar", "DoubleGatePoint", "ThickDoubleGateCar", "ThickDoubleGatePoint", "Obstacle", "Obstacle2", "Road", "Road2d"]:
        for mode in ["dmps", "mps", "td3"]:
            name = f"results/{env}_{mode}_{seed}.log"
            commands.append([
                "python", "main.py",
                "--gpu", None,
                "--env", env,
                "--log_to", name,
                "--mode", mode,
                "--seed", str(seed)
            ])

gpu_ind = 3
parallel_procs = 4
num_free = {0: 4}


if __name__ == "__main__":
    processes = []

    try:
        while True:
            terminated_processes = [(str_cmd, proc) for str_cmd, proc in processes if proc.poll() is not None]
            processes = [(str_cmd, proc) for str_cmd, proc in processes if proc.poll() is None]

            for str_cmd, _ in terminated_processes:
                print(f"FINISHED: {str_cmd}")
                num_free[int(str_cmd.split()[gpu_ind])] += 1

            while len(processes) < parallel_procs and len(commands) != 0:
                # Assign it a gpu
                to_assign = -1
                for gpu in num_free:
                    if num_free[gpu] != 0:
                        num_free[gpu] -= 1
                        to_assign = gpu
                        break

                assert to_assign != -1

                commands[0][gpu_ind] = str(to_assign)

                str_cmd = " ".join(commands[0])
                proc = subprocess.Popen(commands[0], stdout=subprocess.DEVNULL)
                print(f"STARTED: {str_cmd}")
                processes.append((str_cmd, proc))
                commands = commands[1:]

            if len(processes) == 0 and len(commands) == 0:
                break

            time.sleep(10)

    except KeyboardInterrupt:
        print("\n")
        for str_cmd, proc in processes:
            print(f"KILLING: {str_cmd}")
            proc.kill()



