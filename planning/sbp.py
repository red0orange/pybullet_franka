from math import exp, sin, cos

for functor in [
    # simple inequality
    lambda x: (x[1] < x[0] + 1) and (x[1] > x[0] - 1),
    # equation adopted from https://au.mathworks.com/help/matlab/ref/peaks.html
    lambda x: 0
    < (
        3 * (1 - x[0]) ** 2.0 * exp(-(x[0] ** 2) - (x[1] + 1) ** 2)
        - 10 * (x[0] / 5 - x[0] ** 3 - x[1] ** 5) * exp(-x[0] ** 2 - x[1] ** 2)
        - 1 / 3 * exp(-((x[0] + 1) ** 2) - x[1] ** 2)
    ),
    lambda x: -0.22 < (cos(x[0]) * sin(x[1])),
    lambda x: 0.05 < (cos(x[0] ** 2 + x[1] ** 2)),
]:
    engine = sbp_env.engine.BlackBoxEngine(
        collision_checking_functor=functor,
        lower_limits=[-5, -5], upper_limits=[5, 5],
        cc_epsilon=0.1,  # collision check resolution
    )
    planning_args = sbp_env.generate_args(
        planner_id="rrt", engine=engine,
        start_pt=[-3, -3], goal_pt=[3, 3],
        display=True, first_solution=True,
    )

    env = sbp_env.env.Env(args=planning_args)
    env.run()
    print(env.get_solution_path(as_array=True))