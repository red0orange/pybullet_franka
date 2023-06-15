import rospy
from moveit_msgs.msg import PlanningScene, AllowedCollisionMatrix, AllowedCollisionEntry, PlanningSceneComponents, PlanningSceneWorld
from moveit_msgs.srv import GetPlanningScene, ApplyPlanningScene

def allow_collision_and_back():
    rospy.init_node("allow_collision_and_back_node")

    get_planning_scene_service_name = "/my_gen3/get_planning_scene"
    rospy.wait_for_service(get_planning_scene_service_name, 10.0)
    get_planning_scene = rospy.ServiceProxy(get_planning_scene_service_name, GetPlanningScene)
    request = PlanningSceneComponents()
    original_response = get_planning_scene(request)
    ori_planning_scene = original_response.scene
    ori_planning_scene.world = PlanningSceneWorld()

    set_planning_scene_service_name = "/my_gen3/apply_planning_scene"
    rospy.wait_for_service(set_planning_scene_service_name, 10.0)
    apply_planning_scene = rospy.ServiceProxy(set_planning_scene_service_name, ApplyPlanningScene)
    apply_planning_scene(ori_planning_scene)

if __name__ == "__main__":
    allow_collision_and_back()
