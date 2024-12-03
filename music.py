import pygame

pygame.mixer.init()
pygame.mixer.music.stop()

def play_music(music_path):
    if not pygame.mixer.music.get_busy() or pygame.mixer.music.get_pos() == -1:
        pygame.mixer.music.load(music_path)
        pygame.mixer.music.play(-1)

def stop_music():
    pygame.mixer.music.stop()
