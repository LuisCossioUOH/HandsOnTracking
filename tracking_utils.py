import cv2
import numpy as np

class TrackingEllipseGenerator:
    def __init__(self, n_frames, n_tracks, false_positive_rate=0.2, false_negative_rate=0.1,
                 image_shape=[200, 200], duration_samples=[20, 100]):
        self.colors_track = np.random.uniform(0, 255, [n_tracks, 3])
        self.false_negative_rate = false_negative_rate
        self.false_positive_rate = false_positive_rate
        self.n_tracks = n_tracks
        self.n_frames = n_frames
        self.image_shape = image_shape

        self.frame_ids, self.ids_data, self.tracks_data = self.define_coordinates_tracks(duration=duration_samples,
                                                                                         speed_limit=image_shape[
                                                                                                         0] // 10)
        self.detections = []
        self.indices_track = []
        self.frame_names = []
        indices = np.arange(len(self.tracks_data))
        confidence_negative = [0.1, 0.5]
        for j in range(n_frames):
            valid_indx = self.frame_ids == j
            self.indices_track += [indices[valid_indx]]
            subset_tracks = self.tracks_data[valid_indx].reshape([-1, 4])
            subset_detections = self.define_detections_based_tracks(subset_tracks,
                                                                    confidence_negative)
            self.detections += [subset_detections.copy()]

            self.frame_names += ['frame{:d}'.format(j)]

    def define_coordinates_tracks(self, duration=[5, 10], speed_limit=20):
        speeds = np.tile(np.random.randint(-speed_limit, speed_limit, [self.n_tracks, 2]), 2)

        coordinates_start = np.empty([self.n_tracks, 4], dtype=np.int32)
        delta0 = int(self.image_shape[1] * 0.1)
        delta1 = int(self.image_shape[0] * 0.1)
        coordinates_start[:, :2] = np.random.randint([delta1, delta0],
                                                     [self.image_shape[1] - delta0, self.image_shape[0] - delta1],
                                                     [self.n_tracks, 2])

        upper_width = max(delta0, 30)
        upper_height = max(delta1, 30)

        coordinates_start[:, 2:] = coordinates_start[:, :2] + np.random.randint([10, 10], [upper_width, upper_height],
                                                                                [self.n_tracks, 2], dtype=np.int32)
        coordinates_start[:, 0:4:2] = np.clip(coordinates_start[:, 0:4:2], 0, self.image_shape[0])
        coordinates_start[:, 1:4:2] = np.clip(coordinates_start[:, 1:4:2], 0, self.image_shape[1])
        total_annotations = []

        time_spam = np.random.randint(duration[0], min(self.n_frames, duration[1]), [self.n_tracks])
        start_track = np.random.randint(0, self.n_frames - time_spam, [self.n_tracks])
        end_track = start_track + time_spam

        tracks = []
        ids = []
        frames = []
        for j in range(self.n_tracks):
            for i in range(self.n_frames):
                if i >= start_track[j]:
                    # track = coordinates_start[j]
                    # tracks += [track]
                    if i <= end_track[j]:

                        coordinates_start[j] += speeds[j]

                        if coordinates_start[j, 0::2].max() > self.image_shape[1] or coordinates_start[j,
                                                                                     0::2].min() < 0:  # Slowdown exponentially
                            coordinates_start[j, 0::2] -= speeds[j, 0::2]
                            speeds[j, 0::2] = -speeds[j, 0::2] // 2

                        if coordinates_start[j, 1::2].max() > self.image_shape[0] or coordinates_start[j,
                                                                                     1::2].min() < 0:
                            coordinates_start[j, 1::2] -= speeds[j, 1::2]
                            speeds[j, 1::2] = -speeds[j, 1::2] // 2

                        tracks += [coordinates_start[j].copy()]
                        ids += [j]
                        frames += [i]

        final_tracks = np.array(tracks)
        frames = np.array(frames, dtype=np.int32)
        ids = np.array(ids, dtype=np.int32)
        return frames, ids, final_tracks

    def define_detections_based_tracks(self, tracks, confidence_negative=[0.35, 0.85],
                                       lower_confidence_positive=0.8,
                                       n_false_det=3, delta_space=10):
        if len(tracks) > 0:
            coordinates_start = tracks.copy()
            n_tracks = coordinates_start.shape[0]
            wh = (coordinates_start[:, 2:4] - coordinates_start[:, :2]) // 4
            coordinates_start[:, :2] += np.random.randint(-wh, wh, [n_tracks, 2])
            coordinates_start[:, 2:] += np.random.randint(-wh, wh, [n_tracks, 2])
            coordinates_start[:, 0] = np.minimum(coordinates_start[:, 0],
                                                 coordinates_start[:, 2] - delta_space)
            coordinates_start[:, 3] = np.maximum(coordinates_start[:, 3],
                                                 coordinates_start[:, 1] + delta_space)
            confidences = np.random.uniform(lower_confidence_positive, 1.0, len(tracks))

            if np.random.uniform() < self.false_negative_rate:
                delta_dets = min(len(tracks), n_false_det)
                if delta_dets > 1:
                    n_false_negative = np.random.randint(1, delta_dets)
                    n_detections = coordinates_start.shape[0] - n_false_negative
                    indices_keep = np.random.choice(np.arange(coordinates_start.shape[0]), n_detections)
                    coordinates_start = coordinates_start[indices_keep]
                    confidences = confidences[indices_keep]
                else:
                    coordinates_start = np.empty([0, 4])
                    confidences = np.empty([0])

            if np.random.uniform() < self.false_positive_rate:
                delta_dets = max(len(tracks) // 2, n_false_det)
                n_false_positive = np.random.randint(1, delta_dets)
                n_detections = coordinates_start.shape[0] + n_false_positive

                coordinates_start2 = np.empty([n_detections, 4], dtype=np.int32)
                coordinates_start2[:coordinates_start.shape[0]] = coordinates_start

                confidences = np.concatenate([confidences, np.random.uniform(confidence_negative[0],
                                                                             confidence_negative[1], n_false_positive)])

                coordinates_start2[coordinates_start.shape[0]:, :2] = np.random.randint([0, 0], [self.image_shape[1],
                                                                                                 self.image_shape[0]],
                                                                                        [n_false_positive, 2])
                # print("coordinates_start.shape: ",coordinates_start.shape)
                # print("coordinates_start2: ",coordinates_start2.shape)
                # print("n_false_positive: ",n_false_positive)
                coordinates_start2[coordinates_start.shape[0]:, 2:] = coordinates_start2[coordinates_start.shape[0]:,
                                                                      :2] + np.random.randint(6,
                                                                                              self.image_shape[0] // 5,
                                                                                              [n_false_positive, 2],
                                                                                              dtype=np.int32)
                coordinates_start = coordinates_start2.copy()
        else:
            n_detections = np.random.randint(1, n_false_det + 1)
            confidences = np.random.uniform(confidence_negative[0], confidence_negative[1]
                                            , [n_detections])
            coordinates_start = np.empty([n_detections, 4], dtype=np.int32)
            coordinates_start[:, :2] = np.random.randint([0, 0], [self.image_shape[1], self.image_shape[0]],
                                                         [n_detections, 2])
            coordinates_start[:, 2:] = coordinates_start[:, :2] + \
                                       np.random.randint(10, self.image_shape[1] // 3, [n_detections, 2],
                                                         dtype=np.int32)

        confidences = np.reshape(confidences, [-1, 1])
        return np.concatenate([coordinates_start, confidences], axis=1)

    def get_frame(self, idx):
        tracks = self.tracks_data[self.indices_track[idx]]
        img = self.build_image_tracks(tracks, self.ids_data[self.indices_track[idx]])
        return tracks, img

    def get_frame_and_detections(self, idx, draw_detection=False):
        tracks, img = self.get_frame(idx)
        dets = self.detections[idx]
        if draw_detection:
            img = self.build_image_dets(dets, img)
        return tracks, img, dets

    def build_image_tracks(self, tracks, ids_tracks):
        img = np.zeros(self.image_shape + [3], dtype=np.uint8)
        for i, track in enumerate(tracks):
            img = create_images_circles(track, img, self.colors_track[ids_tracks[i]])
        return img

    def build_image_dets(self, dets, img, color=(5, 255, 15)):

        for i, det in enumerate(dets):
            img = create_images_rectangle(det, img, color)
        return img

    def __len__(self):
        return self.n_frames


def create_images_circles(coordinates, image, color, thickness=3):
    center_point = (coordinates[:2] + coordinates[2:]) // 2
    radius = (coordinates[2:] - coordinates[:2]) // 2
    axesLength = (radius[0], radius[1])
    angle = 0
    start_angle = 0
    end_angle = 360
    image = cv2.ellipse(image, center=center_point, axes=axesLength, angle=angle,
                        startAngle=start_angle, endAngle=end_angle, color=color, thickness=thickness)
    return image


def create_images_rectangle(coordinates, image, color, thickness=3):
    # h,w = image_shape[0],image_shape[1]
    point1 = coordinates[:2].astype(np.int32)
    point2 = coordinates[2:4].astype(np.int32)
    image = cv2.rectangle(image, point1, point2, color=color, thickness=thickness)
    return image
