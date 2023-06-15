import rospy
from moveit_msgs.msg import PlanningScene, AllowedCollisionMatrix, AllowedCollisionEntry, PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene, ApplyPlanningScene

def allow_collision_and_back():
    rospy.init_node("allow_collision_and_back_node")

    get_planning_scene_service_name = "/my_gen3/get_planning_scene"
    set_planning_scene_service_name = "/my_gen3/apply_planning_scene"

    rospy.wait_for_service(get_planning_scene_service_name, 10.0)
    get_planning_scene = rospy.ServiceProxy(get_planning_scene_service_name, GetPlanningScene)
    rospy.wait_for_service(set_planning_scene_service_name, 10.0)
    apply_planning_scene = rospy.ServiceProxy(set_planning_scene_service_name, ApplyPlanningScene)

    # Get the original planning scene
    request = PlanningSceneComponents()
    original_response = get_planning_scene(request)
    original_acm = original_response.scene.allowed_collision_matrix

    # Allow all collisions
    acm = original_response.scene.allowed_collision_matrix
    entry = AllowedCollisionEntry(enabled=[True] * len(acm.default_entry.names))
    for name in acm.default_entry.names:
        acm.entry_names.append(name)
        acm.entry_values.append(entry)

    planning_scene_diff = PlanningScene(is_diff=True, allowed_collision_matrix=acm)
    apply_planning_scene(planning_scene_diff)

    # Do your work here (e.g., move arm, grasp object)

    # Revert to the original planning scene
    planning_scene_diff = PlanningScene(is_diff=True, allowed_collision_matrix=original_acm)
    apply_planning_scene(planning_scene_diff)

if __name__ == "__main__":
    allow_collision_and_back()
