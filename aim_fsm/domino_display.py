"""Draw a domino progress display on an existing VEX AIM screen."""

import vex


SCREEN_CENTER_X = 120
# Character advances measured from the VEX AIM Screen API font samples.
MONO24_ADVANCE_PX = 12
MONO20_ADVANCE_PX = 10

PIP_POSITIONS = (
    (),
    ((0, 0),),
    ((-10, -10), (10, 10)),
    ((-10, -10), (0, 0), (10, 10)),
    ((-10, -10), (10, -10), (-10, 10), (10, 10)),
    ((-10, -10), (10, -10), (0, 0), (-10, 10), (10, 10)),
    ((-10, -10), (10, -10), (-10, 0), (10, 0), (-10, 10), (10, 10)),
)


class DominoDisplay:
    """Show short stage names and optional tile progress and pips on the round screen."""

    def __init__(self, screen):
        self.screen = screen
        self._active = False
        self._last_status = None

    def show(self, stage: str, *, tile: tuple[int, int] | None = None,
             pips: tuple[int, int] | None = None, detail: str | None = None,
             color=vex.Color.CYAN) -> bool:
        """Show a stage and a centered detail line, defaulting to the tile count and pips."""
        if self.screen is None:
            return False
        status = (stage, tile, pips, detail, color)
        if status == self._last_status:
            return True

        screen = self.screen
        try:
            if not 1 <= len(stage) <= 12:
                raise ValueError("Stage label must be 1 to 12 characters")
            if detail is not None and len(detail) > 18:
                raise ValueError("Detail line must fit within 18 characters")
            if pips is not None and (len(pips) != 2 or any(not 0 <= value <= 6 for value in pips)):
                raise ValueError("Domino pips must be two values from 0 to 6")

            self._active = True
            if self._last_status is None:
                screen.hide_emoji()
                screen.set_origin(0, 0)
                screen.set_pen_width(2)
                screen.set_fill_color(vex.Color.TRANSPARENT)

            screen.clear_screen(vex.Color.BLACK)
            screen.set_pen_color(color)
            screen.draw_rectangle(100, 32, 40, 80)
            screen.draw_line(100, 72, 140, 72)

            screen.set_pen_color(vex.Color.WHITE)
            if pips is not None:
                for center_y, count in zip((52, 92), pips):
                    for dx, dy in PIP_POSITIONS[count]:
                        screen.draw_circle(120 + dx, center_y + dy, 3, vex.Color.WHITE)

            screen.set_font(vex.FontType.MONO24)
            screen.print_at(stage, x=SCREEN_CENTER_X - len(stage) * MONO24_ADVANCE_PX // 2, y=142)

            if detail is None:
                details = []
                if tile is not None:
                    details.append(f"{tile[0]}/{tile[1]}")
                if pips is not None:
                    details.append(f"{pips[0]} | {pips[1]}")
                detail = "   ".join(details)
            if detail:
                screen.set_font(vex.FontType.MONO20)
                screen.print_at(detail, x=SCREEN_CENTER_X - len(detail) * MONO20_ADVANCE_PX // 2, y=175)
        except Exception as exc:
            self._last_status = None
            print(f"[DOMINO DISPLAY] Unable to draw {stage}: {exc}")
            return False

        self._last_status = status
        return True

    def idle(self) -> None:
        """Restore the built-in smile after this display has used the screen."""
        if not self._active:
            return
        self._active = False
        self._last_status = None
        try:
            self.screen.clear_screen(vex.Color.BLACK)
            self.screen.show_emoji(vex.EmojiType.HAPPY)
        except Exception as exc:
            print(f"[DOMINO DISPLAY] Unable to restore the smile: {exc}")
