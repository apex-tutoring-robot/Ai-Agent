from visuals.chippy_face.chippy_animator import ChippyAnimator
import time

anim = ChippyAnimator()

try:
    while anim.running:
        anim.set_expression("neutral")
        for _ in range(30): anim.draw(); time.sleep(0.05)

        anim.set_expression("happy")
        for _ in range(30): anim.draw(); time.sleep(0.05)

        anim.set_expression("talking")
        for _ in range(30): anim.draw(); time.sleep(0.05)

        anim.set_expression("surprised")
        for _ in range(30): anim.draw(); time.sleep(0.05)

except KeyboardInterrupt:
    anim.close()
