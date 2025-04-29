#!/usr/bin/env bash
#
# ANSI code generator
#
# © Copyright 2015 Tyler Akins
# Licensed under the MIT license with an additional non-advertising clause
# See http://github.com/fidian/ansi

ANSI_ESC=$'\033'
ANSI_CSI="${ANSI_ESC}["
ANSI_OSC="${ANSI_ESC}]"
ANSI_ST="${ANSI_ESC}\\"
ANSI_REPORT="" # The return value from ansi::report

ansi::backward() {
    printf '%s%sD' "$ANSI_CSI" "${1-}"
}

ansi::bell() {
    printf "%s" $'\007'
}

ansi::black() {
    printf '%s30m' "$ANSI_CSI"
}

ansi::blackIntense() {
    printf '%s90m' "$ANSI_CSI"
}

ansi::blink() {
    printf '%s5m' "$ANSI_CSI"
}

ansi::blue() {
    printf '%s34m' "$ANSI_CSI"
}

ansi::blueIntense() {
    printf '%s94m' "$ANSI_CSI"
}

ansi::bgBlack() {
    printf '%s40m' "$ANSI_CSI"
}

ansi::bgBlackIntense() {
    printf '%s100m' "$ANSI_CSI"
}

ansi::bgBlue() {
    printf '%s44m' "$ANSI_CSI"
}

ansi::bgBlueIntense() {
    printf '%s104m' "$ANSI_CSI"
}

ansi::bgColor() {
    printf '%s48;5;%sm' "$ANSI_CSI" "$1"
}

ansi::bgCyan() {
    printf '%s46m' "$ANSI_CSI"
}

ansi::bgCyanIntense() {
    printf '%s106m' "$ANSI_CSI"
}

ansi::bgGreen() {
    printf '%s42m' "$ANSI_CSI"
}

ansi::bgGreenIntense() {
    printf '%s102m' "$ANSI_CSI"
}

ansi::bgMagenta() {
    printf '%s45m' "$ANSI_CSI"
}

ansi::bgMagentaIntense() {
    printf '%s105m' "$ANSI_CSI"
}

ansi::bgRed() {
    printf '%s41m' "$ANSI_CSI"
}

ansi::bgRgb() {
    printf '%s48;2;%s;%s;%sm' "$ANSI_CSI" "$1" "$2" "$3"
}

ansi::bgRedIntense() {
    printf '%s101m' "$ANSI_CSI"
}

ansi::bgWhite() {
    printf '%s47m' "$ANSI_CSI"
}

ansi::bgWhiteIntense() {
    printf '%s107m' "$ANSI_CSI"
}

ansi::bgYellow() {
    printf '%s43m' "$ANSI_CSI"
}

ansi::bgYellowIntense() {
    printf '%s103m' "$ANSI_CSI"
}

ansi::bold() {
    printf '%s1m' "$ANSI_CSI"
}

ansi::color() {
    printf '%s38;5;%sm' "$ANSI_CSI" "$1"
}

ansi::colorCodes() {
    local code i j

    printf 'Standard: '
    ansi::bold
    ansi::white

    for code in 0 1 2 3 4 5 6 7; do
        if [[ "$code" == 7 ]]; then
            ansi::black
        fi
        ansi::colorCodePatch "$code"
    done

    ansi::resetForeground
    ansi::normal
    printf '\nIntense:  '
    ansi::white

    for code in 8 9 10 11 12 13 14 15; do
        if [[ "$code" == 9 ]]; then
            ansi::black
        fi
        ansi::colorCodePatch "$code"
    done

    ansi::resetForeground
    printf '\n\n'

    # for i in 16 22 28 34 40 46; do
    for i in 16 22 28; do
        for j in $i $((i+36)) $((i+72)) $((i+108)) $((i+144)) $((i+180)); do
            ansi::white
            ansi::bold

            for code in $j $((j+1)) $((j+2)) $((j+3)) $((j+4)) $((j+5)); do
                ansi::colorCodePatch "$code"
            done

            ansi::normal
            ansi::resetForeground
            printf '    '
            ansi::black

            for code in $((j+18)) $((j+19)) $((j+20)) $((j+21)) $((j+22)) $((j+23)); do
                ansi::colorCodePatch "$code"
            done

            ansi::resetForeground
            printf '\n'
        done

        printf '\n'
    done

    printf 'Grays:    '
    ansi::bold
    ansi::white

    for code in 232 233 234 235 236 237 238 239 240 241 242 243; do
        ansi::colorCodePatch "$code"
    done

    ansi::resetForeground
    ansi::normal
    printf '\n          '
    ansi::black

    for code in 244 245 246 247 248 249 250 251 252 253 254 255; do
        ansi::colorCodePatch "$code"
    done

    ansi::resetForeground
    printf '\n'
}

ansi::colorCodePatch() {
    ansi::bgColor "$1"
    printf ' %3s ' "$1"
    ansi::resetBackground
}

ansi::colorTable() {
    local colorLabel counter fnbLower fnbUpper functionName IFS resetFunction

    fnbLower="$(
        ansi::faint
        printf f
        ansi::normal
        printf n
        ansi::bold
        printf b
        ansi::normal
        )"
    fnbUpper="$(
        ansi::faint
        printf F
        ansi::normal
        printf N
        ansi::bold
        printf B
        ansi::normal
    )"
    IFS=$' \n'
    counter=

    while read -r colorLabel functionName resetFunction; do
        printf -- '--%s ' "$colorLabel"
        $functionName
        printf 'Sample'
        $resetFunction

        if [[ "$counter" == "x" ]]; then
            counter=
            printf '\n'
        else
            counter=x
            ansi::column 40
        fi
    done <<END
bold ansi::bold ansi::normal
faint ansi::faint ansi::normal
italic ansi::italic ansi::plain
fraktur ansi::fraktur ansi::plain
underline ansi::underline ansi::noUnderline
double-underline ansi::doubleUnderline ansi::noUnderline
blink ansi::blink ansi::noBlink
rapid-blink ansi::rapidBlink ansi::noBlink
inverse ansi::inverse ansi::noInverse
invisible ansi::invisible ansi::visible
strike ansi::strike ansi::noStrike
frame ansi::frame ansi::noBorder
encircle ansi::encircle ansi::noBorder
overline ansi::overline ansi::noOverline
ideogram-right ansi::ideogramRight ansi::resetIdeogram
ideogram-right-double ansi::ideogramRightDouble ansi::resetIdeogram
ideogram-left ansi::ideogramLeft ansi::resetIdeogram
ideogram-left-double ansi::ideogramLeftDouble ansi::resetIdeogram
ideogram-stress ansi::ideogramStress ansi::resetIdeogram
END

    if [[ -n "$counter" ]]; then
        printf '\n'
    fi
    printf '\n'
    printf '            black   red   green  yellow  blue  magenta cyan  white\n'
    ansi::colorTableLine "(none)" "ansi::resetBackground" "$fnbLower" "$fnbUpper"
    ansi::colorTableLine "bg-black" "ansi::bgBlack" "$fnbLower" "$fnbUpper"
    ansi::colorTableLine "+ intense" "ansi::bgBlackIntense" "$fnbLower" "$fnbUpper"
    ansi::colorTableLine "bg-red" "ansi::bgRed" "$fnbLower" "$fnbUpper"
    ansi::colorTableLine "+ intense" "ansi::bgRedIntense" "$fnbLower" "$fnbUpper"
    ansi::colorTableLine "bg-green" "ansi::bgGreen" "$fnbLower" "$fnbUpper"
    ansi::colorTableLine "+ intense" "ansi::bgGreenIntense" "$fnbLower" "$fnbUpper"
    ansi::colorTableLine "bg-yellow" "ansi::bgYellow" "$fnbLower" "$fnbUpper"
    ansi::colorTableLine "+ intense" "ansi::bgYellowIntense" "$fnbLower" "$fnbUpper"
    ansi::colorTableLine "bg-blue" "ansi::bgBlue" "$fnbLower" "$fnbUpper"
    ansi::colorTableLine "+ intense" "ansi::bgBlueIntense" "$fnbLower" "$fnbUpper"
    ansi::colorTableLine "bg-magenta" "ansi::bgMagenta" "$fnbLower" "$fnbUpper"
    ansi::colorTableLine "+ intense" "ansi::bgMagentaIntense" "$fnbLower" "$fnbUpper"
    ansi::colorTableLine "bg-cyan" "ansi::bgCyan" "$fnbLower" "$fnbUpper"
    ansi::colorTableLine "+ intense" "ansi::bgCyanIntense" "$fnbLower" "$fnbUpper"
    ansi::colorTableLine "bg-white" "ansi::bgWhite" "$fnbLower" "$fnbUpper"
    ansi::colorTableLine "+ intense" "ansi::bgWhiteIntense" "$fnbLower" "$fnbUpper"

    printf '\n'
    printf 'Legend:\n'
    printf '    Normal color:  f = faint, n = normal, b = bold.\n'
    printf '    Intense color:  F = faint, N = normal, B = bold.\n'
}

ansi::colorTableLine() {
    local fn

    printf '%-12s' "$1"
    for fn in ansi::black ansi::red ansi::green ansi::yellow ansi::blue ansi::magenta ansi::cyan ansi::white; do
        $2
        ${fn}
        printf '%s' "$3"
        ${fn}Intense
        printf '%s' "$4"
        ansi::resetForeground
        ansi::resetBackground

        if [[ "$fn" != "ansi::white" ]]; then
            printf ' '
        fi
    done
    printf '\n'
}

ansi::column() {
    printf '%s%sG' "$ANSI_CSI" "${1-}"
}

ansi::columnRelative() {
    printf '%s%sa' "$ANSI_CSI" "${1-}"
}

ansi::cyan() {
    printf '%s36m' "$ANSI_CSI"
}

ansi::cyanIntense() {
    printf '%s96m' "$ANSI_CSI"
}

ansi::deleteChars() {
    printf '%s%sP' "$ANSI_CSI" "${1-}"
}

ansi::deleteLines() {
    printf '%s%sM' "$ANSI_CSI" "${1-}"
}

ansi::doubleUnderline() {
    printf '%s21m' "$ANSI_CSI"
}

ansi::down() {
    printf '%s%sB' "$ANSI_CSI" "${1-}"
}

ansi::encircle() {
    printf '%s52m' "$ANSI_CSI"
}

ansi::eraseDisplay() {
    printf '%s%sJ' "$ANSI_CSI" "${1-}"
}

ansi::eraseChars() {
    printf '%s%sX' "$ANSI_CSI" "${1-}"
}

ansi::eraseLine() {
    printf '%s%sK' "$ANSI_CSI" "${1-}"
}

ansi::faint() {
    printf '%s2m' "$ANSI_CSI"
}

ansi::font() {
    printf '%s1%sm' "$ANSI_CSI" "${1-0}"
}

ansi::forward() {
    printf '%s%sC' "$ANSI_CSI" "${1-}"
}

ansi::fraktur() {
    printf '%s20m' "$ANSI_CSI"
}

ansi::frame() {
    printf '%s51m' "$ANSI_CSI"
}

ansi::green() {
    printf '%s32m' "$ANSI_CSI"
}

ansi::greenIntense() {
    printf '%s92m' "$ANSI_CSI"
}

ansi::hideCursor() {
    printf '%s?25l' "$ANSI_CSI"
}

ansi::ideogramLeft() {
    printf '%s62m' "$ANSI_CSI"
}

ansi::ideogramLeftDouble() {
    printf '%s63m' "$ANSI_CSI"
}

ansi::ideogramRight() {
    printf '%s60m' "$ANSI_CSI"
}

ansi::ideogramRightDouble() {
    printf '%s61m' "$ANSI_CSI"
}

ansi::ideogramStress() {
    printf '%s64m' "$ANSI_CSI"
}

ansi::insertChars() {
    printf '%s%s@' "$ANSI_CSI" "${1-}"
}

ansi::insertLines() {
    printf '%s%sL' "$ANSI_CSI" "${1-}"
}

ansi::inverse() {
    printf '%s7m' "$ANSI_CSI"
}

ansi::invisible() {
    printf '%s8m' "$ANSI_CSI"
}

ansi::isAnsiSupported() {
    # Optionally override detection logic
    # to support post processors that interpret color codes _after_ output is generated.
    # Use environment variable "ANSI_FORCE_SUPPORT=<anything>" to enable the override.
    if [[ -n "${ANSI_FORCE_SUPPORT-}" ]]; then
        return 0
    fi

    if hash tput &> /dev/null; then
        if [[ "$(tput colors)" -lt 8 ]]; then
            return 1
        fi

        return 0
    fi

    # Query the console and see if we get ANSI codes back.
    # CSI 0 c == CSI c == Primary Device Attributes.
    # Idea:  CSI c
    # Response = CSI ? 6 [234] ; 2 2 c
    # The "22" means ANSI color, but terminals don't need to send that back.
    # If we get anything back, let's assume it works.
    ansi::report c "$ANSI_CSI?" c || return 1
    [[ -n "$ANSI_REPORT" ]]
}

ansi::italic() {
    printf '%s3m' "$ANSI_CSI"
}

ansi::line() {
    printf '%s%sd' "$ANSI_CSI" "${1-}"
}

ansi::lineRelative() {
    printf '%s%se' "$ANSI_CSI" "${1-}"
}

ansi::magenta() {
    printf '%s35m' "$ANSI_CSI"
}

ansi::magentaIntense() {
    printf '%s95m' "$ANSI_CSI"
}

ansi::nextLine() {
    printf '%s%sE' "$ANSI_CSI" "${1-}"
}

ansi::noBlink() {
    printf '%s25m' "$ANSI_CSI"
}

ansi::noBorder() {
    printf '%s54m' "$ANSI_CSI"
}

ansi::noInverse() {
    printf '%s27m' "$ANSI_CSI"
}

ansi::normal() {
    printf '%s22m' "$ANSI_CSI"
}

ansi::noOverline() {
    printf '%s55m' "$ANSI_CSI"
}

ansi::noStrike() {
    printf '%s29m' "$ANSI_CSI"
}

ansi::noUnderline() {
    printf '%s24m' "$ANSI_CSI"
}

ansi::overline() {
    printf '%s53m' "$ANSI_CSI"
}

ansi::plain() {
    printf '%s23m' "$ANSI_CSI"
}

ansi::position() {
    local position="${1-}"
    printf '%s%sH' "$ANSI_CSI" "${position/,/;}"
}

ansi::previousLine() {
    printf '%s%sF' "$ANSI_CSI" "${1-}"
}

ansi::rapidBlink() {
    printf '%s6m' "$ANSI_CSI"
}

ansi::red() {
    printf '%s31m' "$ANSI_CSI"
}

ansi::redIntense() {
    printf '%s91m' "$ANSI_CSI"
}

ansi::repeat() {
    printf '%s%sb' "$ANSI_CSI" "${1-}"
}

ansi::report() {
    local buff c report

    report=""

    # Note: read bypass piping, which lets this work:
    # ansi --report-window-chars | cut -d , -f 1
    read -p "$ANSI_CSI$1" -r -N "${#2}" -s -t 1 buff

    if [ "$buff" != "$2" ]; then
        return 1
    fi

    read -r -N "${#3}" -s -t 1 buff

    while [[ "$buff" != "$3" ]]; do
        report="$report${buff:0:1}"
        read -r -N 1 -s -t 1 c || exit 1
        buff="${buff:1}$c"
    done

    ANSI_REPORT=$report
}

ansi::reportPosition() {
    ansi::report 6n "$ANSI_CSI" R || return 1
    printf '%s\n' "${ANSI_REPORT//;/,}"
}

ansi::reportIcon() {
    ansi::report 20t "${ANSI_OSC}L" "$ANSI_ST" || return 1
    printf '%s\n' "${ANSI_REPORT//;/,}"
}

ansi::reportScreenChars() {
    ansi::report 19t "${ANSI_CSI}9;" t || return 1
    printf '%s\n' "${ANSI_REPORT//;/,}"
}

ansi::reportTitle() {
    ansi::report 21t "${ANSI_OSC}l" "$ANSI_ST" || return 1
    printf '%s\n' "${ANSI_REPORT//;/,}"
}

ansi::reportWindowChars() {
    ansi::report 18t "${ANSI_CSI}8;" t || return 1
    printf '%s\n' "${ANSI_REPORT//;/,}"
}

ansi::reportWindowPixels() {
    ansi::report 14t "${ANSI_CSI}4;" t || return 1
    printf '%s\n' "${ANSI_REPORT//;/,}"
}

ansi::reportWindowPosition() {
    ansi::report 13t "${ANSI_CSI}3;" t || return 1
    printf '%s\n' "${ANSI_REPORT//;/,}"
}

ansi::reportWindowState() {
    ansi::report 11t "$ANSI_CSI" t || return 1
    case "$ANSI_REPORT" in
        1)
            printf 'open\n'
            ;;

        2)
            printf 'iconified\n'
            ;;

        *)
            printf 'unknown (%s)\n' "$ANSI_REPORT"
            ;;
    esac
}

ansi::reset() {
    ansi::resetColor
    ansi::resetFont
    ansi::eraseDisplay 2
    ansi::position "1;1"
    ansi::showCursor
}

ansi::resetAttributes() {
    printf '%s22;23;24;25;27;28;29;54;55m' "$ANSI_CSI"
}

ansi::resetBackground() {
    printf '%s49m' "$ANSI_CSI"
}

ansi::resetColor() {
    printf '%s0m' "$ANSI_CSI"
}

ansi::resetFont() {
    printf '%s10m' "$ANSI_CSI"
}

ansi::resetForeground() {
    printf '%s39m' "$ANSI_CSI"
}

ansi::resetIdeogram() {
    printf '%s65m' "$ANSI_CSI"
}

ansi::restoreCursor() {
    printf '%su' "$ANSI_CSI"
}

ansi::rgb() {
    printf '%s38;2;%s;%s;%sm' "$ANSI_CSI" "$1" "$2" "$3"
}

ansi::saveCursor() {
    printf '%ss' "$ANSI_CSI"
}

ansi::scrollDown() {
    printf '%s%sT' "$ANSI_CSI" "${1-}"
}

ansi::scrollUp() {
    printf '%s%sS' "$ANSI_CSI" "${1-}"
}

ansi::icon() {
    printf '%s1;%s%s' "$ANSI_OSC" "${1-}" "$ANSI_ST"
}

ansi::title() {
    printf '%s2;%s%s' "$ANSI_OSC" "${1-}" "$ANSI_ST"
}

ansi::showCursor() {
    printf '%s?25h' "$ANSI_CSI"
}

ansi::showHelp() {
    cat <<EOF
Generate ANSI escape codes

Please keep in mind that your terminal must support the code in order for you
to see the effect properly.

Usage
    ansi [OPTIONS] [TEXT TO OUTPUT]

Option processing stops at the first unknown option or at "--".  Options
are applied in order as specified on the command line.  Unless --no-restore
is used, the options are unapplied in reverse order, restoring your
terminal to normal.

Optional parameters are surrounded in brackets and use reasonable defaults.
For instance, using --down will move the cursor down one line and --down=10
moves the cursor down 10 lines.

Display Manipulation
    --insert-chars[=N], --insert-char[=N], --ich[=N]
                             Insert blanks at cursor, shifting the line right.
    --erase-display[=N], --ed[=N]
                             Erase in display. 0=below, 1=above, 2=all,
                             3=saved.
    --erase-line=[N], --el[=N]
                             Erase in line. 0=right, 1=left, 2=all.
    --insert-lines[=N], --insert-line[=N], --il[=N]
    --delete-lines[=N], --delete-line[=N], --dl[=N]
    --delete-chars[=N], --delete-char[=N], --dch[=N]
    --scroll-up[=N], --su[=N]
    --scroll-down[=N], --sd[=N]
    --erase-chars[=N], --erase-char[=N], --ech[=N]
    --repeat[=N], --rep[=N]  Repeat preceding character N times.

Cursor:
    --up[=N], --cuu[=N]
    --down[=N], --cud[=N]
    --forward[=N], --cuf[=N]
    --backward[=N], --cub[=N]
    --next-line[=N], --cnl[=N]
    --previous-line[=N], --prev-line[=N], --cpl[=N]
    --column[=N], --cha[=N]
    --position[=[ROW],[COL]], --cup[=[ROW],[=COL]]
    --tab-forward[=N]        Move forward N tab stops.
    --tab-backward[=N]       Move backward N tab stops.
    --column-relative[=N], --hpr[=N]
    --line[=N], --vpa[=N]
    --line-relative[=N], --vpr[=N]
    --save-cursor            Saves the cursor position.  Restores the cursor
                             after writing text to the terminal unless
                             --no-restore is also used.
    --restore-cursor         Just restores the cursor.
    --hide-cursor            Will automatically show cursor unless --no-restore
                             is also used.
    --show-cursor

Text:
    Attributes:
        --bold, --faint, --normal
        --italic, --fraktur, --plain
        --underline, --double-underline, --no-underline
        --blink, --rapid-blink, --no-blink
        --inverse, --no-inverse
        --invisible, --visible
        --strike, --no-strike
        --frame, --encircle, --no-border
        --overline, --no-overline
        --ideogram-right, --ideogram-right-double, --ideogram-left,
        --ideogram-left-double, --ideogram-stress, --reset-ideogram
        --font=NUM (NUM must be from 0 through 9 and 0 is the primary font)
    Foreground:
        --black, --red, --green, --yellow, --blue, --magenta, --cyan, --white,
        --black-intense, --red-intense, --green-intense, --yellow-intense,
        --blue-intense, --magenta-intense, --cyan-intense, --white-intense,
        --color=CODE, --rgb=R,G,B
    Background:
        --bg-black, --bg-red, --bg-green, --bg-yellow, --bg-blue,
        --bg-magenta, --bg-cyan, --bg-white, --bg-black-intense,
        --bg-red-intense, --bg-green-intense, --bg-yellow-intense,
        --bg-blue-intense, --bg-magenta-intense, --bg-cyan-intense,
        --bg-white-intense, --bg-color=CODE, --bg-rgb=R,G,B
    Reset:
        --reset-attrib       Removes bold, italics, etc.
        --reset-foreground   Sets foreground to default color.
        --reset-background   Sets background to default color.
        --reset-color        Resets attributes, foreground, background.
        --reset-font         Switches back to the primary font.

Report:
    ** NOTE:  These require reading from stdin.  Results are sent to stdout.
    ** If no response from terminal in 1 second, these commands fail.
    --report-position        ROW,COL
    --report-window-state    "open" or "iconified"
    --report-window-position X,Y
    --report-window-pixels   HEIGHT,WIDTH
    --report-window-chars    ROWS,COLS
    --report-screen-chars    ROWS,COLS of the entire screen
    --report-icon
    --report-title

Miscellaneous:
    --color-table            Display a color table.
    --color-codes            Show the different color codes.
    --icon=NAME              Set the terminal's icon name.
    --title=TITLE            Set the terminal's window title.
    --no-restore             Do not issue reset codes when changing colors.
                             For example, if you change the color to green,
                             normally the color is restored to default
                             afterwards.  With this flag, the color will
                             stay green even when the command finishes.
    -n, --no-newline         Suppress newline at the end of the line.
    --bell                   Add the terminal's bell sequence to the output.
    --reset                  Reset colors, clear screen, show cursor, move
                             cursor to (1,1), and reset the font.
EOF
}

ansi::strike() {
    printf '%s9m' "$ANSI_CSI"
}

ansi::tabBackward() {
    printf '%s%sZ' "$ANSI_CSI" "${1-}"
}

ansi::tabForward() {
    printf '%s%sI' "$ANSI_CSI" "${1-}"
}

ansi::underline() {
    printf '%s4m' "$ANSI_CSI"
}

ansi::up() {
    printf '%s%sA' "$ANSI_CSI" "${1-}"
}

ansi::visible() {
    printf '%s28m' "$ANSI_CSI"
}

ansi::white() {
    printf '%s37m' "$ANSI_CSI"
}

ansi::whiteIntense() {
    printf '%s97m' "$ANSI_CSI"
}

ansi::yellow() {
    printf '%s33m' "$ANSI_CSI"
}

ansi::yellowIntense() {
    printf '%s93m' "$ANSI_CSI"
}

ansi() {
    local addNewline b g m r readOptions restoreText restoreCursorPosition restoreCursorVisibility supported triggerRestore
    local m10 m22 m23 m24 m25 m27 m28 m29 m39 m49 m54 m55 m65

    addNewline=true
    m10=
    m22=
    m23=
    m24=
    m25=
    m27=
    m28=
    m29=
    m39=
    m49=
    m54=
    m55=
    m65=
    readOptions=true
    restoreCursorPosition=false
    restoreCursorVisibility=false
    restoreText=false
    supported=true
    triggerRestore=true

    if ! ansi::isAnsiSupported; then
        supported=false
    fi

    while $readOptions && [[ $# -gt 0 ]]; do
        case "$1" in
            --help | -h | -\?)
                ansi::showHelp
                return 0
                ;;

            # Display Manipulation
            --insert-chars | --insert-char | --ich)
                $supported && ansi::insertChars
                ;;

            --insert-chars=* | insert-char=* | --ich=*)
                $supported && ansi::insertChars "${1#*=}"
                ;;

            --erase-display | --ed)
                $supported && ansi::eraseDisplay
                ;;

            --erase-display=* | --ed=*)
                $supported && ansi::eraseDisplay "${1#*=}"
                ;;

            --erase-line | --el)
                $supported && ansi::eraseLine
                ;;

            --erase-line=* | --el=*)
                $supported && ansi::eraseLine "${1#*=}"
                ;;

            --insert-lines | --insert-line | --il)
                $supported && ansi::insertLines
                ;;

            --insert-lines=* | --insert-line=* | --il=*)
                $supported && ansi::insertLines "${1#*=}"
                ;;

            --delete-lines | --delete-line | --dl)
                $supported && ansi::deleteLines
                ;;

            --delete-lines=* | --delete-line=* | --dl=*)
                $supported && ansi::deleteLines "${1#*=}"
                ;;

            --delete-chars | --delete-char | --dch)
                $supported && ansi::deleteChars
                ;;

            --delete-chars=* | --delete-char=* | --dch=*)
                $supported && ansi::deleteChars "${1#*=}"
                ;;

            --scroll-up | --su)
                $supported && ansi::scrollUp
                ;;

            --scroll-up=* | --su=*)
                $supported && ansi::scrollUp "${1#*=}"
                ;;

            --scroll-down | --sd)
                $supported && ansi::scrollDown
                ;;

            --scroll-down=* | --sd=*)
                $supported && ansi::scrollDown "${1#*=}"
                ;;

            --erase-chars | --erase-char | --ech)
                $supported && ansi::eraseChars
                ;;

            --erase-chars=* | --erase-char=* | --ech=*)
                $supported && ansi::eraseChars "${1#*=}"
                ;;

            --repeat | --rep)
                $supported && ansi::repeat
                ;;

            --repeat=* | --rep=N)
                $supported && ansi::repeat "${1#*=}"
                ;;

            # Cursor Positioning
            --up | --cuu)
                $supported && ansi::up
                ;;

            --up=* | --cuu=*)
                $supported && ansi::up "${1#*=}"
                ;;

            --down | --cud)
                $supported && ansi::down
                ;;

            --down=* | --cud=*)
                $supported && ansi::down "${1#*=}"
                ;;

            --forward | --cuf)
                $supported && ansi::forward
                ;;

            --forward=* | --cuf=*)
                $supported && ansi::forward "${1#*=}"
                ;;

            --backward | --cub)
                $supported && ansi::backward
                ;;

            --backward=* | --cub=*)
                $supported && ansi::backward "${1#*=}"
                ;;

            --next-line | --cnl)
                $supported && ansi::nextLine
                ;;

            --next-line=* | --cnl=*)
                $supported && ansi::nextLine "${1#*=}"
                ;;

            --previous-line | --prev-line | --cpl)
                $supported && ansi::previousLine
                ;;

            --previous-line=* | --prev-line=* | --cpl=*)
                $supported && ansi::previousLine "${1#*=}"
                ;;

            --column | --cha)
                $supported && ansi::column
                ;;

            --column=* | --cha=*)
                $supported && ansi::column "${1#*=}"
                ;;

            --position | --cup)
                $supported && ansi::position
                ;;

            --position=* | --cup=*)
                $supported && ansi::position "${1#*=}"
                ;;

            --tab-forward | --cht)
                $supported && ansi::tabForward
                ;;

            --tab-forward=* | --cht=*)
                $supported && ansi::tabForward "${1#*=}"
                ;;

            --tab-backward | --cbt)
                $supported && ansi::tabBackward
                ;;

            --tab-backward=* | --cbt=*)
                $supported && ansi::tabBackward "${1#*=}"
                ;;

            --column-relative | --hpr)
                $supported && ansi::columnRelative
                ;;

            --column-relative=* | --hpr=*)
                $supported && ansi::columnRelative "${1#*=}"
                ;;

            --line | --vpa)
                $supported && ansi::line
                ;;

            --line=* | --vpa=*)
                $supported && ansi::line "${1#*=}"
                ;;

            --line-relative | --vpr)
                $supported && ansi::lineRelative
                ;;

            --line-relative=* | --vpr=*)
                $supported && ansi::lineRelative "${1#*=}"
                ;;

            --save-cursor)
                $supported && ansi::saveCursor
                restoreCursorPosition=true
                ;;

            --restore-cursor)
                $supported && ansi::restoreCursor
                ;;

            --hide-cursor)
                $supported && ansi::hideCursor
                restoreCursorVisibility=true
                ;;

            --show-cursor)
                $supported && ansi::showCursor
                ;;

            # Colors - Attributes
            --bold)
                $supported && ansi::bold
                restoreText=true
                m22="22;"
                ;;

            --faint)
                $supported && ansi::faint
                restoreText=true
                m22="22;"
                ;;

            --italic)
                $supported && ansi::italic
                restoreText=true
                m23="23;"
                ;;

            --underline)
                $supported && ansi::underline
                restoreText=true
                m24="24;"
                ;;

            --blink)
                $supported && ansi::blink
                restoreText=true
                m25="25;"
                ;;

            --rapid-blink)
                $supported && ansi::rapidBlink
                restoreText=true
                m25="25;"
                ;;

            --inverse)
                $supported && ansi::inverse
                restoreText=true
                m27="27;"
                ;;

            --invisible)
                $supported && ansi::invisible
                restoreText=true
                m28="28;"
                ;;

            --strike)
                $supported && ansi::strike
                restoreText=true
                m29="29;"
                ;;

            --font|--font=0)
                $supported && ansi::resetFont
                ;;

            --font=[123456789])
                $supported && ansi::font "${1#*=}"
                restoreText=true
                m10="10;"
                ;;

            --fraktur)
                $supported && ansi::fraktur
                restoreText=true
                m23="23;"
                ;;

            --double-underline)
                $supported && ansi::doubleUnderline
                restoreText=true
                m24="24;"
                ;;

            --normal)
                $supported && ansi::normal
                ;;

            --plain)
                $supported && ansi::plain
                ;;

            --no-underline)
                $supported && ansi::noUnderline
                ;;

            --no-blink)
                $supported && ansi::noBlink
                ;;

            --no-inverse)
                $supported && ansi::noInverse
                ;;

            --visible)
                $supported && ansi::visible
                ;;

            --no-strike)
                $supported && ansi::noStrike
                ;;

            --frame)
                $supported && ansi::frame
                restoreText=true
                m54="54;"
                ;;

            --encircle)
                $supported && ansi::encircle
                restoreText=true
                m54="54;"
                ;;

            --overline)
                $supported && ansi::overline
                restoreText=true
                m55="55;"
                ;;

            --no-border)
                $supported && ansi::noBorder
                ;;

            --no-overline)
                $supported && ansi::noOverline
                ;;

            # Colors - Foreground
            --black)
                $supported && ansi::black
                restoreText=true
                m39="39;"
                ;;

            --red)
                $supported && ansi::red
                restoreText=true
                m39="39;"
                ;;

            --green)
                $supported && ansi::green
                restoreText=true
                m39="39;"
                ;;

            --yellow)
                $supported && ansi::yellow
                restoreText=true
                m39="39;"
                ;;

            --blue)
                $supported && ansi::blue
                restoreText=true
                m39="39;"
                ;;

            --magenta)
                $supported && ansi::magenta
                restoreText=true
                m39="39;"
                ;;

            --cyan)
                $supported && ansi::cyan
                restoreText=true
                m39="39;"
                ;;

            --white)
                $supported && ansi::white
                restoreText=true
                m39="39;"
                ;;

            --black-intense)
                $supported && ansi::blackIntense
                restoreText=true
                m39="39;"
                ;;

            --red-intense)
                $supported && ansi::redIntense
                restoreText=true
                m39="39;"
                ;;

            --green-intense)
                $supported && ansi::greenIntense
                restoreText=true
                m39="39;"
                ;;

            --yellow-intense)
                $supported && ansi::yellowIntense
                restoreText=true
                m39="39;"
                ;;

            --blue-intense)
                $supported && ansi::blueIntense
                restoreText=true
                m39="39;"
                ;;

            --magenta-intense)
                $supported && ansi::magentaIntense
                restoreText=true
                m39="39;"
                ;;

            --cyan-intense)
                $supported && ansi::cyanIntense
                restoreText=true
                m39="39;"
                ;;

            --white-intense)
                $supported && ansi::whiteIntense
                restoreText=true
                m39="39;"
                ;;

            --rgb=*,*,*)
                r=${1#*=}
                b=${r##*,}
                g=${r#*,}
                g=${g%,*}
                r=${r%%,*}
                $supported && ansi::rgb "$r" "$g" "$b"
                restoreText=true
                m39="39;"
                ;;

            --color=*)
                $supported && ansi::color "${1#*=}"
                restoreText=true
                m39="39;"
                ;;

            # Colors - Background
            --bg-black)
                $supported && ansi::bgBlack
                restoreText=true
                m49="49;"
                ;;

            --bg-red)
                $supported && ansi::bgRed
                restoreText=true
                m49="49;"
                ;;

            --bg-green)
                $supported && ansi::bgGreen
                restoreText=true
                m49="49;"
                ;;

            --bg-yellow)
                $supported && ansi::bgYellow
                restoreText=true
                m49="49;"
                ;;

            --bg-blue)
                $supported && ansi::bgBlue
                restoreText=true
                m49="49;"
                ;;

            --bg-magenta)
                $supported && ansi::bgMagenta
                restoreText=true
                m49="49;"
                ;;

            --bg-cyan)
                $supported && ansi::bgCyan
                restoreText=true
                m49="49;"
                ;;

            --bg-white)
                $supported && ansi::bgWhite
                restoreText=true
                m49="49;"
                ;;

            --bg-black-intense)
                $supported && ansi::bgBlackIntense
                restoreText=true
                m49="49;"
                ;;

            --bg-red-intense)
                $supported && ansi::bgRedIntense
                restoreText=true
                m49="49;"
                ;;

            --bg-green-intense)
                $supported && ansi::bgGreenIntense
                restoreText=true
                m49="49;"
                ;;

            --bg-yellow-intense)
                $supported && ansi::bgYellowIntense
                restoreText=true
                m49="49;"
                ;;

            --bg-blue-intense)
                $supported && ansi::bgBlueIntense
                restoreText=true
                m49="49;"
                ;;

            --bg-magenta-intense)
                $supported && ansi::bgMagentaIntense
                restoreText=true
                m49="49;"
                ;;

            --bg-cyan-intense)
                $supported && ansi::bgCyanIntense
                restoreText=true
                m49="49;"
                ;;

            --bg-white-intense)
                $supported && ansi::bgWhiteIntense
                restoreText=true
                m49="49;"
                ;;

            --bg-rgb=*,*,*)
                r=${1#*=}
                b=${r##*,}
                g=${r#*,}
                g=${g%,*}
                r=${r%%,*}
                $supported && ansi::bgRgb "$r" "$g" "$b"
                restoreText=true
                m49="49;"
                ;;

            --bg-color=*)
                $supported && ansi::bgColor "${1#*=}"
                restoreText=true
                m49="49;"
                ;;

            # Colors - Reset
            --reset-attrib)
                $supported && ansi::resetAttributes
                ;;

            --reset-foreground)
                $supported && ansi::resetForeground
                ;;

            --reset-background)
                $supported && ansi::resetBackground
                ;;

            --reset-color)
                $supported && ansi::resetColor
                ;;

            --reset-font)
                $supported && ansi::resetFont
                ;;

            # Reporting
            --report-position)
                $supported || return 1
                ansi::reportPosition || return $?
                ;;

            --report-window-state)
                $supported || return 1
                ansi::reportWindowState || return $?
                ;;

            --report-window-position)
                $supported || return 1
                ansi::reportWindowPosition || return $?
                ;;

            --report-window-pixels)
                $supported || return 1
                ansi::reportWindowPixels || return $?
                ;;

            --report-window-chars)
                $supported || return 1
                ansi::reportWindowChars || return $?
                ;;

            --report-screen-chars)
                $supported || return 1
                ansi::reportScreenChars || return $?
                ;;

            --report-icon)
                $supported || return 1
                ansi::reportIcon || return $?
                ;;

            --report-title)
                $supported || return 1
                ansi::reportTitle || return $?
                ;;

            --ideogram-right)
                $supported && ansi::ideogramRight
                restoreText=true
                m65="65;"
                ;;

            --ideogram-right-double)
                $supported && ansi::ideogramRightDouble
                restoreText=true
                m65="65;"
                ;;

            --ideogram-left)
                $supported && ansi::ideogramLeft
                restoreText=true
                m65="65;"
                ;;

            --ideogram-left-double)
                $supported && ansi::ideogramLeftDouble
                restoreText=true
                m65="65;"
                ;;

            --ideogram-stress)
                $supported && ansi::ideogramStress
                restoreText=true
                m65="65;"
                ;;

            --reset-ideogram)
                $supported && ansi::noIdeogram
                ;;

            # Miscellaneous
            --color-codes)
                if ! $supported; then
                    echo "ANSI is not supported in this terminal."
                else
                    ansi::colorCodes
                fi

                return 0
                ;;

            --color-table)
                if ! $supported; then
                    echo "ANSI is not supported in this terminal."
                else
                    ansi::colorTable
                fi

                return 0
                ;;

            --icon)
                $supported && ansi::icon ""
                ;;

            --icon=*)
                $supported && ansi::icon "${1#*=}"
                ;;

            --title)
                $supported && ansi::title ""
                ;;

            --title=*)
                $supported && ansi::title "${1#*=}"
                ;;

            --no-restore)
                triggerRestore=false
                ;;

            -n | --no-newline)
                addNewline=false
                ;;

            --bell)
                ansi::bell
                ;;

            --reset)
                $supported || return 0
                ansi::reset
                ;;

            --)
                readOptions=false
                shift
                ;;

            *)
                readOptions=false
                ;;
        esac

        if $readOptions; then
            shift
        fi
    done

    printf '%s' "${1-}"

    if [[ "$#" -gt 1 ]]; then
        shift || :
        printf "${IFS:0:1}%s" "${@}"
    fi

    if $supported && $triggerRestore; then
        if $restoreCursorPosition; then
            ansi::restoreCursor
        fi

        if $restoreCursorVisibility; then
            ansi::showCursor
        fi

        if $restoreText; then
            m="$m10$m22$m23$m24$m25$m27$m28$m29$m39$m49$m54$m55$m65"
            printf '%s%sm' "$ANSI_CSI" "${m%;}"
        fi
    fi

    if $addNewline; then
        printf '\n'
    fi
}


# Run if not sourced
if [[ "$0" == "${BASH_SOURCE[0]}" ]]; then
    ansi "$@" || exit $?
fi
