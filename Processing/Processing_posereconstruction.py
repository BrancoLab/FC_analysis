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

# TODO class: skeleton visualiser (plot skeleton and size/angle of stuff)
# TODO class: bp visualiser (position and velocity of bp over time), + ...
# the same for connection and maybe for whole sk reg

from Utils.maths import *
from Plotting.Plotting_utils import *


class Skeleton:
    def __init__(self, frame_num):
        self.name = None
        self.metadata = None
        self.frame_num = frame_num

        # vars to facilitate writing stuff
        lear, snout, rear, neck = 'lear', 'snout', 'rear', 'neck'
        body, base, t1, t2 = 'body', 'tail', 't1', 't2'

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
                                right_back=(rear, neck), left_back=(neck, lear))
        body_connections = dict(front_body=(neck, body), back_body=(body, base))
        tail_connections = dict(first_segmend=(base, t1),second_segment=(t1, t2))
        self.all_connections = dict(head=head_connections, body=body_connections, tail=tail_connections)

        self.head_connections = {c: None for c in head_connections.keys()}
        self.body_connections = {c: None for c in body_connections.keys()}
        self.tail_connections = {c: None for c in tail_connections.keys()}

    def __repr__(self):
        return 'Skeleton - frame {}'.format(self.frame_num)

    def __getattr__(self, item):
        return self.__dict__[item]


class BodyPart:
    def __init__(self, position=None):
        self.name = None
        self.skeleton_region = None
        self.connected_bparts = None
        self.position = position

    def __repr__(self):
        return 'Bodypart: {}'.format(self.name)


class Bone:
    def __init__(self):
        self.name = None
        self.connected_bparts = None
        self.orientation = None
        self.ang_velocity = None
        self.length = None
        self.prev_orientation = None

    def determine_length(self):
        self.length = calc_distance_2d((self.connected_bparts[0].position, self.connected_bparts[1].position),
                                       vectors=False)

    def determine_orientation(self):
        self.orientation = calc_angle_2d(self.connected_bparts[0].position, self.connected_bparts[1].position)

    def determine_ang_vel(self):
        if self.prev_orientation is None:
            warnings.warn('Could not calculate ang_vel as prev. orientation is None')
            self.ang_velocity = 0

        if self.orientation is None:
            self.determine_orientation()

        self.ang_velocity = self.orientation - self.prev_orientation

    def __repr__(self):
        return  'Bone connecting {} to {}'.format(self.connected_bparts[0].name, self.connected_bparts[1].name)


class Constructor:
    def __init__(self, frame_num, session_metadata, trial_metadata, frame_tracking_data, prev_skeleton=None):
        """  Given the posture data for a frame (from DLC), construt the frame """

        # TODO define names and metadatas
        self.skeleton = Skeleton(frame_num)

        # Create skeleton and each body part, assign to skeleton
        for sk_region, bodyparts in self.skeleton.all_parts.items():
            for bp in bodyparts:
                if bp not in frame_tracking_data.keys():
                    use_bp = bp.title()
                    if use_bp not in frame_tracking_data.keys():
                        a = 1
                        continue
                else: use_bp = bp
                # Create the BodyPart instance and populate
                pos = (frame_tracking_data[use_bp]['x'], frame_tracking_data[use_bp]['y'])
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

                bp_objects, regions = [], ['head', 'body', 'tail']
                for bp in bodyparts:
                    for reg_name in regions:
                        reg = getattr(self.skeleton, reg_name)

                        if not bp in reg.keys(): continue
                        else:
                            bp_objects.append(reg[bp])

                if None in bp_objects: continue

                connection.connected_bparts = tuple(bp_objects)
                connection.determine_length()
                connection.determine_orientation()
                self.skeleton.__dict__['{}_connections'.format(sk_region)][con_name] = connection


class SkeletonPlotter:
    def __init__(self, skeleton, ax=None, adjust_ax_lim=False, verbose=False):
        if ax is None:
            f, ax = create_figure()
        ax.set(facecolor=[.2, .2, .2], title=skeleton.name)

        colors = dict(
            head=[.2, .8, .1],
            body=[.5, .4, .8],
            tail=[.7, .2, .2]
        )

        for region_name, color in colors.items():
            reg = getattr(skeleton, region_name)
            for name, bp in reg.items():
                if bp is None: continue
                if bp.name == 'body' and adjust_ax_lim:
                    xlim = [bp.position[0] - 25, bp.position[0] + 25]
                    ylim = [bp.position[1] - 25, bp.position[1] + 25]
                    ax.set(xlim=xlim, ylim=ylim)
                ax.scatter(bp.position[0], bp.position[1], color=color)
                if verbose:
                    ax.text(bp.position[0]-2, bp.position[1]-2, bp.name, color=color, fontsize=8)

            reg_con = getattr(skeleton, '{}_connections'.format(region_name))
            for name, bparts in reg_con.items():
                if bparts is None: continue

                ax.plot([bparts.connected_bparts[0].position[0], bparts.connected_bparts[1].position[0]],
                        [bparts.connected_bparts[0].position[1], bparts.connected_bparts[1].position[1]],
                        color=color, linewidth=2)

                if verbose:
                    x = bparts.connected_bparts[0].position[0] + \
                        ((bparts.connected_bparts[1].position[0] - bparts.connected_bparts[0].position[0]))/2
                    y = bparts.connected_bparts[0].position[1] + \
                        ((bparts.connected_bparts[1].position[1] - bparts.connected_bparts[0].position[1])) / 2
                    test_pos = (x, y-1)
                    ax.text(test_pos[0], test_pos[1], name, color=color, fontsize=6)
                    ax.text(test_pos[0], test_pos[1]-.3, 'Len: {}'.format(round(bparts.length, 2)), color=color, fontsize=6)
                    ax.text(test_pos[0], test_pos[1]-.6, 'Ori: {}'.format(round(bparts.orientation, 2)),
                            color=color, fontsize=6)


class OverseeConstruction:
    """  get tracking data, loop over frames - call classes and handle output  """
    def __init__(self, session_metadata=None, trial_metadata=None, data=None, display=False, display_many=False):
        print('Extracting posture from DLC data')
        dlc_posture = data.dlc_tracking['Posture']
        nframes = len(list(dlc_posture.values())[0])

        # Get list of bodyparts present in both model and data
        lear, snout, rear, neck = 'lear', 'snout', 'rear', 'neck'
        body, base, t1, t2 = 'body', 'tail', 't1', 't2'
        model_parts = [lear, snout, rear, neck, body, base, t1, t2]
        tracking_parts = [c for c in dlc_posture.keys() if c.lower() in model_parts]

        self.skeletons = []
        for f in tqdm(range(nframes)):
            position_data = {bp: dlc_posture[bp].iloc[f] for bp in tracking_parts}

            if f == 0:
                prev_sk = None
            else:
                prev_sk = self.skeletons[f-1]
            constr = Constructor(f, session_metadata, trial_metadata, position_data, prev_skeleton=prev_sk)

            if display:
                SkeletonPlotter(constr.skeleton)

            self.skeletons.append(constr.skeleton)

        if display_many:
            n_subplots = np.ceil(nframes/100)
            n_cols = np.int(np.ceil(n_subplots/4))

            # f, axarr = create_figure(ncols=n_cols, nrows=4)
            # axarr = axarr.flatten()
            #
            # for i, ax in enumerate(axarr):
            #     SkeletonPlotter(skeletons[i*100], ax, adjust_ax_lim=False)

            f, ax = create_figure()
            for i, skeleton in enumerate(self.skeletons):
                if i % 50 == 0: SkeletonPlotter(skeleton, ax, adjust_ax_lim=False)

    def calculate_body_length(self, visualise=True):
        """  Calculates the total length of the mouse main axis """

        axis_connections = dict(
            head=['main_axis'],
            body=['front_body', 'back_body'],
            tail=['first_segmend', 'second_segment']
        )

        axis_lengths, fake_axis_lengths = [], []
        head_main, head_side = [], []
        for skeleton in self.skeletons:
            accumulator = 0
            for region, conn_names in axis_connections.items():
                for conn_name in conn_names:
                    connection = skeleton.__dict__['{}_connections'.format(region)][conn_name]
                    if connection is None: continue
                    accumulator += connection.length
            axis_lengths.append(accumulator)

            snout = skeleton.head['snout']
            tail = skeleton.tail['tail']
            fake_axis_lengths.append(calc_distance_2d((snout.position, tail.position), vectors=False))

            head_main.append(skeleton.head_connections['main_axis'].length)
            head_side.append(skeleton.head_connections['ear_axis'].length)


        if visualise:
            f, axarr = create_figure(nrows=2)

            axarr[0].plot(np.divide(axis_lengths, axis_lengths[1799]), linewidth=2, alpha=.7, label='body axis')
            axarr[0].plot(np.divide(fake_axis_lengths, fake_axis_lengths[1799]), linewidth=2, alpha=.7, label='snout axis')
            axarr[1].plot(np.divide(head_main, head_main[1799]), linewidth=2, alpha=.7, label='main head')
            axarr[1].plot(np.divide(head_side, head_side[1799]), linewidth=2, alpha=.7, label='side head')

            for ax in axarr:
                ax.set(xlim=[1775, 1830], facecolor=[.2, .2, .2])
                ax.axvline(1800, color='w')
                make_legend(ax, changefont=8)

            plt.show()


        a = 1
































