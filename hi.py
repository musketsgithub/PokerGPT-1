import sdl  # Or PySDL2, depending on how you import

def main():
    if sdl.SDL_Init(sdl2.SDL_INIT_VIDEO) < 0:
        print("SDL initialization failed:", sdl2.SDL_GetError())
        return 1

    # ... your SDL code ...

    # Incorrect (Illustrative - Check PySDL2 docs for correct way):
    # app = ...  # However you're getting your SDL object
    # version = app.macOSVersion()  # This will likely cause the error

    # Correct (Illustrative - Check PySDL2 docs, it might not be directly available. You might need to use platform-specific Python modules):
    # import platform
    # print(platform.system())  # Get OS name
    # print(platform.mac_ver()) # Get macOS version (if on macOS)

    sdl2.SDL_Quit()
    return 0

if __name__ == "__main__":
    main()