import vlc
import time

player = vlc.MediaPlayer("video.mp4")

player.play()

time.sleep(10)

player.stop()