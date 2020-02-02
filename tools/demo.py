# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
import shutil
import os
import datetime
import subprocess
from tools.test import *
import numpy as np

parser = argparse.ArgumentParser(description="PyTorch Tracking Demo")

parser.add_argument(
    "--resume",
    default="",
    type=str,
    required=True,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--config",
    dest="config",
    default="config_davis.json",
    help="hyper-parameter of SiamMask in json format",
)
parser.add_argument("--base_path", default="../../data/tennis", help="datasets")
parser.add_argument("--cpu", action="store_true", help="cpu mode")
parser.add_argument(
    "--writevid", action="store_true", help="save output as a video (requires ffmpeg)"
)
args = parser.parse_args()

if __name__ == "__main__":
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    from custom import Custom

    siammask = Custom(anchors=cfg["anchors"])
    if args.resume:
        assert isfile(args.resume), "Please download {} first.".format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    # Parse Image file
    image_dirname = os.path.basename(args.base_path)
    img_files = sorted(glob.glob(join(args.base_path, "*.jp*")))
    ims = [cv2.imread(imf) for imf in img_files]

    # Set output dir
    timestamp = "{:%Y%m%d-%H%M%S}".format(datetime.datetime.now())
    output_dir = os.path.join("./output", f"{image_dirname}_{timestamp}")
    output_frame_dir = os.path.join(output_dir, "frames")
    output_video_path = os.path.join(output_dir, "output.mp4")
    # Create dir
    os.makedirs(output_frame_dir, exist_ok=True)

    # define initial rectangles per test video
    init_rects = {
        "tennis": [(300, 110, 165, 250)],
        "bream-test-a": [(378, 682, 220, 146)],
        "bream-test-b": [(1810, 538, 108, 92), (464, 660, 264, 124)],
        "bream-test-c": [(739, 595, 87, 90), (1243, 433, 203, 86)],
    }
    # select pre-defined initial rectangles
    init_rects = init_rects[image_dirname]

    active_trackers = []

    toc = 0
    for f, im in enumerate(ims):
        tic = cv2.getTickCount()
        # copy input image for output
        image_out = im.copy()
        if f == 0:  # init
            for index, (x, y, w, h) in enumerate(init_rects):
                target_pos = np.array([x + w / 2, y + h / 2])
                target_sz = np.array([w, h])
                tracker = siamese_init(
                    im, target_pos, target_sz, siammask, cfg["hp"], device=device
                )  # init tracker
                active_trackers.append(
                    {"state": tracker, "score_history": [], "active": True}
                )
        elif f > 0:  # tracking
            for index, tracker in enumerate(active_trackers):
                if not tracker["active"]:
                    continue

                state = siamese_track(
                    tracker["state"],
                    im,
                    mask_enable=True,
                    refine_enable=True,
                    device=device,
                )  # track
                location = state["ploygon"].flatten()
                mask = state["mask"] > state["p"].seg_thr
                score = state["score"]

                tracker["state"] = state
                tracker["score_history"].append(score)

                # check tracker score history
                if len(tracker["score_history"]) >= 3:
                    last_scores = tracker["score_history"][-3:]
                    # if last scores are not greater than threshold, deactivate tracker
                    if not all([score > 0.8 for score in last_scores]):
                        tracker["active"] = False
                        continue

                # write polyline and score to new image
                # cannot write direct to image, as being processed multiple times
                image_out[:, :, 2] = (mask > 0) * 155 + (mask == 0) * image_out[:, :, 2]
                rect_points = [np.int0(location).reshape((-1, 1, 2))]
                cv2.polylines(image_out, rect_points, True, (0, 255, 0), 1)
                minX = min([point[0][0] for point in rect_points[0]])
                minY = min([point[0][1] for point in rect_points[0]])
                cv2.putText(
                    image_out,
                    str(round(score, 2)),
                    (minX, minY),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                )

            filepath = os.path.join(output_frame_dir, str(f).zfill(5) + ".jpg")
            cv2.imwrite(filepath, image_out)

        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print(
        "SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)".format(
            toc, fps
        )
    )

    if args.writevid:
        print("Writing output to video", output_video_path)
        # requires ffmpeg
        shell_command = f"ffmpeg -hide_banner -r 20 -f image2 -i {output_frame_dir}/%5d.jpg -crf 4 -vcodec libx264 -pix_fmt yuv420p {output_video_path}".split(
            " "
        )
        subprocess.run(shell_command)
