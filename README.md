# interview_meter
 Measure display time for interviewer and guest

Original idea came while watching Joe Rogan's interview with Lex Fridman
                    (https://www.youtube.com/watch?v=g4OJooMbgRE&t=7624s)

I thought: this guy Rogan talks more than the guest!

Then I did the math, and it happened to be true. This script is what I used

Instructions:

Extract the reference frames using ffmpeg

ffmpeg -ss 00:00:XX -t 00:00:00.01 -i YOURMOVIE.MP4 -r 25.0 REFERENCE_NAME.jpg

Replace XX with the exact second where the scene is displaying
(from https://stackoverflow.com/questions/8287759/extracting-frames-from-mp4-flv)

You have to extract every reference image and leave it in the directory

Now run the script with the argument -f VIDEOFILE.mp4

The script generates simple stats and a file VIDEOFILE.csv to further analyze stats

Please see the Notebook in the project to analyze and create plots
