"""
Series of classes to organise the processing of DLC tracking data to reconstruct the pose of the mouse.
This should facilitate stuff like detection time and reaction time identification

Hirarchy:

- Skeleton:
    - head:
        - snout
        - lear
        - rear
        - neck
    - body:
        - body centre
    - tail
        - tail base
        - tail 1...
        - tail n


Classes:
* Skeleton:
    - name (e.g. frame num)
    - metadata (session, animal, trial...)

* Bodypart:
    - name
    - skeleton region
    - connections
    - position

* Bone:
    - name
    - connecting body parts
    - orientation
    - ang velocity
    - length

"""

class Skeleton:
    def __init__(self):
        self.name = None
        self.metadata = None

        # vars to facilitate writing stuff
        lear, snout, rear, neck = 'lear', 'snout', 'rear', 'neck'
        body, base, t1, t2 = 'body', 'base', 't1', 't2'


        # Body parts
        head_parts = [lear, snout, rear, neck]
        body_parts = [body]
        tail_parts = [base, t1, t2]
        self.all_parts = dict(head=head_parts, body=body_parts, tail=tail_parts)

        self.head = {p: None for p in head_parts}
        self.body = {p: None for p in body_parts}
        self.tail = {p: None for p in tail_parts}

        # Connections
        head_connections = dict(ear_axis=(lear, rear),
                                main_axis=(snout, neck),
                                left_front=(lear, snout), right_front=(snout, rear),
                                right_back=(rear, neck), left_back=(neck, rear))
        body_connections = dict(front_body=(neck, body), back_body=(body, base))
        tail_connections = dict(first_segmend=(base, t1),second_segment=(t1, t2))
        self.all_connections = dict(head=head_connections, body=body_connections, tail=tail_connections)

        self.head_connections = {c: None for c in head_connections.keys()}
        self.body_connections = {c: None for c in body_connections.keys()}
        self.tail_connections = {c: None for c in tail_connections.keys()}


class BodyPart:
    def __init__(self, position=None):
        self.name = None
        self.skeleton_region = None
        self.connected_bparts = None
        self.position = position


class Bone:
    def __init__(self):
        self.name = None
        self.connected_bparts = None
        self.orientation = None
        self.ang_velocity = None
        self.length = None

    def determine_length(self):
        pass

    def determine_orientation(self):
        pass

    def determine_ang_vel(self):
        pass


class Constructor:
    def __init__(self, session_metadata, trial_metadata, frame_tracking_data, prev_skeleton=None):
        """  Given the posture data for a frame (from DLC), construt the frame """

        # TODO define names and metadatas
        self.skeleton = Skeleton()

        # Create skeleton and each body part, assign to skeleton
        for sk_region, bodyparts in self.skeleton.all_parts.items():
            for bp in bodyparts:
                # Create the BodyPart instance and populate
                pos = (frame_tracking_data[bp]['x'], frame_tracking_data[bp]['y'])
                part = BodyPart(position=pos)
                part.name, part.skeleton_region = bp, sk_region

                # Assign to skeleton
                skeleton_region = getattr(self.skeleton, sk_region)
                skeleton_region[bp] = part

        # Create connections
        for sk_region, connections in self.skeleton.all_connections.items():
            for con_name, bodyparts in connections.items():
                connection = Bone()
                connection.name = con_name
                skeleton_region = getattr(self.skeleton, sk_region)
                connection.connected_bparts = (getattr(skeleton_region, bodyparts[0]),
                                               getattr(skeleton_region, bodyparts[1]))


class OverseeConstruction:
    """  get tracking data, loop over frames - call classes and handle output  """
    def __init__(self, data=None):
        dlc_posture = data.dlc_tracking['Posture']
        nframes = len(dlc_posture)

        # Get list of bodyparts
        lear, snout, rear, neck = 'lear', 'snout', 'rear', 'neck'
        body, base, t1, t2 = 'body', 'base', 't1', 't2'
        model_parts = [lear, snout, rear, neck, body, base, t1, t2]

        tracking_parts = [c for c in dlc_posture.columns if c in model_parts]

        for f in range(nframes):
            position_data = {bp:dlc_posture[bp].iloc[f] for bp in tracking_parts}








# TODO class: skeleton visualiser (plot skeleton and size/angle of stuff)
# TODO class: bp visualiser (position and velocity of bp over time), + the same for connection and maybe for whole sk reg





































