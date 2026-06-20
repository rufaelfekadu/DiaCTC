
from .text import preprocess_text


def get_diacritic_class(idx, line, case_ending, arabic_letters, diacritic_classes):
    # Handle without case ending
    if not case_ending:
        end = True
        for i in range(idx + 1, len(line)):
            if line[i] not in diacritic_classes:
                end = line[i].isspace()
                break
        if end:
            return -1

    if idx + 1 >= len(line) or line[idx + 1] not in diacritic_classes:
        # No diacritic
        return 0

    diac = line[idx + 1]

    if idx + 2 >= len(line) or line[idx + 2] not in diacritic_classes:
        # Only one diacritic
        return diacritic_classes.index(diac) + 1

    diac += line[idx + 2]

    try:
        # Try the possibility of double diacritics
        return diacritic_classes.index(diac) + 1
    except ValueError:
        try:
            # Try the possibility of reversed double diacritics
            return diacritic_classes.index(diac[::-1]) + 1
        except ValueError:
            # Otherwise consider only the first diacritic
            return diacritic_classes.index(diac[0]) + 1


def get_diacritics_classes(line, case_ending, arabic_letters, diacritic_classes, style):
    classes = list()
    for idx, char in enumerate(line):
        if style == "Fadel":
            if char in arabic_letters:
                classes.append(
                    get_diacritic_class(
                        idx, line, case_ending, arabic_letters, diacritic_classes
                    )
                )
        elif style == "Zitouni":
            if char in diacritic_classes or char.isspace():
                continue
            classes.append(
                get_diacritic_class(
                    idx, line, case_ending, arabic_letters, diacritic_classes
                )
            )
    return classes


def clear_line(line, arabic_letters, diacritic_classes):
    line = preprocess_text(line)
    line = " ".join(
        "".join(
            [
                (
                    char
                    if char in list(arabic_letters) + diacritic_classes + [" "]
                    else " "
                )
                for char in line
            ]
        ).split()
    )
    new_line = ""
    for idx, char in enumerate(line):
        if char not in diacritic_classes or (idx > 0 and line[idx - 1] != " "):
            new_line += char
    line = new_line
    new_line = ""
    for idx, char in enumerate(line):
        if char not in diacritic_classes or (idx > 0 and line[idx - 1] != " "):
            new_line += char
    return new_line


def extract_base_characters(line, arabic_letters, diacritic_classes, style):
    """
    Extract base characters (Arabic letters) from a line, preserving their positions.
    Returns a list of tuples: (position_in_line, base_char)
    """
    base_chars = []
    for idx, char in enumerate(line):
        if style == "Fadel":
            if char in arabic_letters:
                base_chars.append((idx, char))
        elif style == "Zitouni":
            if char not in diacritic_classes and not char.isspace():
                base_chars.append((idx, char))
    return base_chars


def align_chars_dp(original_chars, target_chars):
    """
    Use dynamic programming to align character sequences and count operations.
    Returns (substitutions, deletions, insertions, total_reference_chars, alignment)
    where alignment is a list of tuples: (orig_idx, target_idx) with None for insertions/deletions.
    """
    m = len(original_chars)
    n = len(target_chars)

    # DP table: dp[i][j] = minimum edit distance
    dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0

    # Fill DP table
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                continue

            if i > 0 and j > 0:
                # Character comparison - characters are tuples (position, char)
                if original_chars[i - 1][1] == target_chars[j - 1][1]:
                    # Match - no cost
                    dp[i][j] = min(dp[i][j], dp[i - 1][j - 1])
                else:
                    # Substitution
                    dp[i][j] = min(dp[i][j], dp[i - 1][j - 1] + 1)

            if i > 0:
                # Deletion
                dp[i][j] = min(dp[i][j], dp[i - 1][j] + 1)

            if j > 0:
                # Insertion
                dp[i][j] = min(dp[i][j], dp[i][j - 1] + 1)

    # Backtrack to build alignment and count operations
    substitutions = 0
    deletions = 0
    insertions = 0
    alignment = []

    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            if original_chars[i - 1][1] == target_chars[j - 1][1]:
                # Match
                alignment.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i - 1][j - 1] + 1:
                # Substitution
                alignment.append((i - 1, j - 1))
                substitutions += 1
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i - 1][j] + 1:
                # Deletion
                alignment.append((i - 1, None))
                deletions += 1
                i -= 1
            else:
                # Insertion
                alignment.append((None, j - 1))
                insertions += 1
                j -= 1
        elif i > 0:
            # Deletion
            alignment.append((i - 1, None))
            deletions += 1
            i -= 1
        else:
            # Insertion
            alignment.append((None, j - 1))
            insertions += 1
            j -= 1

    # Reverse alignment since we built it backwards
    alignment.reverse()

    return substitutions, deletions, insertions, m, alignment


def get_aligned_diacritic_classes(
    original_line, target_line, alignment, original_base_chars, target_base_chars,
    case_ending, arabic_letters, diacritic_classes
):
    """
    Get diacritic classes for aligned positions only.
    Returns two lists: (original_classes, target_classes) for aligned positions.
    """
    original_classes_aligned = []
    target_classes_aligned = []

    for orig_pos, target_pos in alignment:
        if orig_pos is not None and target_pos is not None:
            # Both positions exist, get diacritic classes
            if orig_pos < len(original_base_chars) and target_pos < len(target_base_chars):
                orig_char_pos = original_base_chars[orig_pos][0]
                target_char_pos = target_base_chars[target_pos][0]

                orig_class = get_diacritic_class(
                    orig_char_pos, original_line, case_ending, arabic_letters, diacritic_classes
                )
                target_class = get_diacritic_class(
                    target_char_pos, target_line, case_ending, arabic_letters, diacritic_classes
                )

                original_classes_aligned.append(orig_class)
                target_classes_aligned.append(target_class)
            else:
                # Position out of bounds, skip
                continue
        elif orig_pos is not None:
            # Deletion in target - count as error
            original_classes_aligned.append(-2)  # Special marker for deletion
            target_classes_aligned.append(-3)  # Special marker for missing
        elif target_pos is not None:
            # Insertion in target - count as error
            original_classes_aligned.append(-3)  # Special marker for missing
            target_classes_aligned.append(-2)  # Special marker for insertion

    return original_classes_aligned, target_classes_aligned
