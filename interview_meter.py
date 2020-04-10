#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
  interview_meter.py: Analyze camera time for studio interviews.
  Author: Joaquin Cabezas github.com/joaquincabezas
  Date: 10/04/2020

  Instructions:

    Extract the reference frames using ffmpeg
    ffmpeg -ss 00:00:XX -t 00:00:00.01 -i YOURMOVIE.MP4 -r 25.0 REFERENCE_NAME.jpg
    Replace XX with the exact second where the scene is displaying
    (from https://stackoverflow.com/questions/8287759/extracting-frames-from-mp4-flv)

    You have to extract every reference image and leave it in the directory

    Now run the script with the argument -f VIDEOFILE.mp4

    The script generates simple stats and a file VIDEOFILE.csv to further analyze stats
    Please see the Notebook in the project to analyze and create plots

"""

import argparse
import glob
import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def scenes():
    """ Gather the different scenes available to compare each video frame

    Args:
        None
    Return:
        scenarios (dict): The images of each scenario prepared to be compared
        scenarios_name (dict): The index and name of each scenario
    """

    scenarios = {}
    scenes_name = {}
    index = 0

    # loop over the images we already extracted (see instructions)
    for image_path in glob.glob("./*.png"):

        filename = image_path[image_path.rfind("/") + 1:]
        image = cv2.imread(image_path)

        # We strip out the "./" and the extension of the file to get just the name
        name = os.path.splitext(filename)[0][2:]

        # For SSIM we use grayscale images
        scenarios[name] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # We create a dictionary with the indexes to use just numbers in numpy array
        scenes_name[index] = name
        index += 1

    return scenarios, scenes_name

def stats(project_name, matches, scenes_name):
    """ Creates the statistics and outputs some simple stats

    Args:
        project_name (str): Name of the video we are processing
        matches (numpy array): Array containing which scene is displayed in each second
        scenarios_name (dict): Names of the different scenarios
    Return:
        None
    """
    np.savetxt(project_name + '.csv', matches, delimiter=',', fmt='%1d')

    print("Simple stats. Display time:")
    for idx, name in scenes_name.items():
        percentage = round(100*np.count_nonzero(matches == idx)/len(matches))
        print('For ' + name + ': ' + str(percentage) + '%')


def open_video(input_file):
    """ Open the video file and returns basic information

    Args:
        input_file (str): Route to the video file
    Return:
        cap (VideoCapture) OpenCV2 object
        total_frames (int): Number of frames of the video (we will be using less, only one per sec)
        frames_per_sec (int): Frame rate
    """
    cap = cv2.VideoCapture(input_file)
    frames_per_sec = round(cap.get(5)) #frame rate
    total_frames = round(cap.get(7))

    return cap, total_frames, frames_per_sec

def main():
    """ Main function

    Prepare the input data and go through a loop for every frame in the video
    Compared every key_frame (1 per sec) to the different scenes and prepare an array
    Then call stats to save this information for future use

    """

    # Get arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True)
    parser.add_argument('-s', '--show')
    args = parser.parse_args()

    input_file = args.file
    show_video = args.show

    project_name = input_file[0:-4]

    # Open the input file video and return also the total number of frames
    cap, total_frames, frames_per_sec = open_video(input_file)

    # We prepare an array with every keyframe (1 frame per second)
    # We use numpy so you can develop any statistics study easily
    total_keyframes = round(total_frames/frames_per_sec)

    matches = np.zeros(total_keyframes, dtype=np.int32)

    scenarios, scenes_name = scenes()
    results = {}

    keyframe_count = 0
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()

        # We evaluate only one frame per second
        is_key_frame = (frame_count % frames_per_sec) == 0

        # We analyze each key frame
        if ret and is_key_frame:
            video_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # We compare ech scenario with the current keyframe
            for key, scenario_image in scenarios.items():
                score, _diff = ssim(scenario_image, video_image, full=True)
                results[key] = score

            # We select the most probable match and add to the array in the current keyframe
            max_result = max(results, key=results.get)
            max_result_idx = list(scenes_name.keys())[list(scenes_name.values()).index(max_result)]
            matches[keyframe_count] = max_result_idx

            # We can show the video to check the analysis is working properly
            if show_video:
                cv2.putText(frame, matches[keyframe_count],
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
                cv2.imshow('Frame', frame)

                # And finish it at any time by pressing 'q' (typical for CV)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            keyframe_count += 1

            if (keyframe_count % round(total_keyframes/20)) == 0:
                print_progress(keyframe_count, total_keyframes, 'Progress:', 'Complete', length=70)

        # Once we reach the end of the video, we exit the loop
        if keyframe_count == total_keyframes:
            break

        # In case not every keyframe is available, we exit once arriving at the end of the video
        if frame_count == total_frames:
            break

        frame_count += 1
    # When everything is done, release the video capture object

    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    stats(project_name, matches, scenes_name)

def print_progress(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console

    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    progress_bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, progress_bar, percent, suffix))
    # Print New Line on Complete
    if iteration == total:
        print()

if __name__ == "__main__":
    main()
