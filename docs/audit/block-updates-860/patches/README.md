# Rejected experimental patches

These reproduce the evaluated candidates against frozen baseline
`eb74a914d779371a02e4a6637eaa724aeb66fdf5`. None met the acceptance gate.
They are research artifacts, not changes to the package implementation.

Apply only inside a separately archived source directory with `patch -p1`.
Each patch only appends value-write methods; it changes no existing method,
including whole-iterate state copying. The inlined and outlined variants have
identical copy/activity algorithms; their only executable difference is the
helper's inlining annotation. The singleton variant retains Base for longer
buffers and custom index descriptors.
