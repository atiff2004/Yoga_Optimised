import cv2
import mediapipe as mp
import math
import time

class YogaAnalyzer:
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.body_turned = False
        self.hands_gripped = False
        self.start_time_count = None
        self.calory_burned = 0

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=False, min_detection_confidence=self.detectionCon)

        # This dictionary will hold the results for each analysis
        self.results_dict = {}

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img, draw=True):
        lmlist = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return lmlist

    def find_middle_point(self, x1, y1, x2, y2):
        middle_x = (x1 + x2) // 2
        middle_y = (y1 + y2) // 2
        return int(middle_x), int(middle_y)

    def calculate_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def calculate_angle(self, x1, y1, x2, y2, x3, y3):
        vector1 = [x1 - x2, y1 - y2]
        vector2 = [x3 - x2, y3 - y2]
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
        magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
        cosine_angle = dot_product / (magnitude1 * magnitude2)
        cosine_angle = max(min(cosine_angle, 1.0), -1.0)  # Ensure within range
        return math.degrees(math.acos(cosine_angle))

    def calculate_collinearity_percentage(self, x1, y1, x2, y2, x3, y3):
        vector1 = [x2 - x1, y2 - y1]
        vector2 = [x3 - x1, y3 - y1]
        cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]
        epsilon = 1e-6
        if abs(cross_product) < epsilon:
            return 100.0
        return (1 - abs(cross_product) / (abs(vector1[0] * vector2[1]) + abs(vector1[1] * vector2[0]))) * 100

    def calculate_overall_progress(self, front_angle, back_angle, hand_body_angle):
        thresholds = [83, 83, 130]
        progress_scores = [
            min(front_angle / thresholds[0] * 100, 100),
            min(back_angle / thresholds[1] * 100, 100),
            min(hand_body_angle / thresholds[2] * 100, 100),
        ]
        return sum(progress_scores) / 3

    def calculate_calories_burned(self, elapsed_time, progress_score):
        # Define a realistic calorie burn rate per minute for yoga
        base_calorie_burn_rate_per_minute = 4  # Example: 4 calories per minute
        # Calculate calories burned based on elapsed time and progress score
        adjusted_progress_score = progress_score / 100  # Convert to 0-1 scale
        calories_burned = (elapsed_time / 60) * base_calorie_burn_rate_per_minute * adjusted_progress_score
        return calories_burned

    def analyze_pose(self, img):
        img = self.findPose(img)
        lmlist = self.getPosition(img, draw=False)

        if len(lmlist) == 0:
            # Reset if no landmarks detected
            self.start_time_count = None
            self.calory_burned = 0
            self.results_dict.clear()
            return img

        # Calculate key points
        Head_Mid_point_x, Head_Mid_point_y = self.find_middle_point(lmlist[5][1], lmlist[5][2], lmlist[2][1], lmlist[2][2])
        Hand_Mid_point_x, Hand_Mid_point_y = self.find_middle_point(lmlist[16][1], lmlist[16][2], lmlist[15][1], lmlist[15][2])
        Body_Mid_point_x, Body_Mid_point_y = self.find_middle_point(lmlist[24][1], lmlist[24][2], lmlist[23][1], lmlist[23][2])

        lef_leg_distance = self.calculate_distance(Body_Mid_point_x, Body_Mid_point_y, lmlist[27][1], lmlist[27][2])
        right_leg_distance = self.calculate_distance(Body_Mid_point_x, Body_Mid_point_y, lmlist[28][1], lmlist[28][2])

        base_point_x, base_point_y = (lmlist[27][1], lmlist[27][2]) if lef_leg_distance > right_leg_distance else (lmlist[28][1], lmlist[28][2])
        back_point_x, back_point_y = (lmlist[28][1], lmlist[28][2]) if lef_leg_distance > right_leg_distance else (lmlist[27][1], lmlist[27][2])

        hands_close_distance = self.calculate_distance(lmlist[16][1], lmlist[16][2], lmlist[15][1], lmlist[15][2])
        self.hands_gripped = hands_close_distance < 30
        
        body_mid_points_distance = self.calculate_distance(lmlist[24][1], lmlist[24][2], lmlist[23][1], lmlist[23][2])
        self.body_turned = body_mid_points_distance < 30

        # Valid pose check
        if not self.body_turned:
            return img  # Do not calculate if body is not turned

        front_angle = self.calculate_angle(Head_Mid_point_x, Head_Mid_point_y, Body_Mid_point_x, Body_Mid_point_y, base_point_x, base_point_y)
        back_angle = self.calculate_angle(base_point_x, base_point_y, Body_Mid_point_x, Body_Mid_point_y, back_point_x, back_point_y)

        # Check if the pose is valid
        if 90 <= front_angle <= 105 and 75 <= back_angle <= 80:
            self.start_time_count = None  # Reset the timer if in resting position
            self.calory_burned = 0
            self.results_dict.clear()
            return img

        # Calculate results
        body_line_percentage = self.calculate_collinearity_percentage(Hand_Mid_point_x, Hand_Mid_point_y, Body_Mid_point_x, Body_Mid_point_y, back_point_x, back_point_y)
        hand_body_angle = self.calculate_angle(Hand_Mid_point_x, Hand_Mid_point_y, Head_Mid_point_x, Head_Mid_point_y, Body_Mid_point_x, Body_Mid_point_y)
        yoga_progress = self.calculate_overall_progress(front_angle, back_angle, hand_body_angle)

        # Time and calorie calculation
        if self.start_time_count is None and 40 < yoga_progress < 80:
            self.start_time_count = time.time()

        if self.start_time_count is not None:
            elapsed_seconds = int(time.time() - self.start_time_count)
            self.calory_burned += self.calculate_calories_burned(elapsed_seconds, yoga_progress)
            self.results_dict = {
                "front_angle": round(front_angle, 2),  # Round for easier reading
                "back_angle": round(back_angle, 2),
                "body_line_percentage": round(body_line_percentage, 2),
                "hand_body_angle": round(hand_body_angle, 2),
                "yoga_progress": round(yoga_progress, 2),
                "calories_burned": round(self.calory_burned, 2),  # Round for easier reading
                "elapsed_time": elapsed_seconds,
                "hands_gripped_status": "Hands close to each other" if self.hands_gripped else "Hands not close to each other",
                "body_turned_status": "Body turned" if self.body_turned else "Body not turned",
            }

        return img

    def get_results(self):
        return self.results_dict
