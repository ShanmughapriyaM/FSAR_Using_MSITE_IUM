class SkeletonVelocityExtractor:
    def __init__(self):
        self.pose = None
        self.prev_pose = None
        self.num_joints = 33

    def _init_pose(self):
        if self.pose is None:
            import mediapipe as mp
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

    def reset(self):
        self.prev_pose = None

    def extract(self, frame):
        self._init_pose()   

        import cv2
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame)

        if results.pose_landmarks is None:
            pose = np.zeros(self.num_joints * 2, dtype=np.float32)
        else:
            pose = np.zeros(self.num_joints * 2, dtype=np.float32)
            for i, lm in enumerate(results.pose_landmarks.landmark):
                pose[2*i]   = lm.x
                pose[2*i+1] = lm.y

        if self.prev_pose is None:
            velocity = np.zeros_like(pose)
        else:
            velocity = pose - self.prev_pose

        self.prev_pose = pose.copy()
        return np.concatenate([pose, velocity])
