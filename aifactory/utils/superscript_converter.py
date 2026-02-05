class SuperscriptConverter:

    # Unicode: superscript map
    SUPERSCRIPT_MAP = {
        # digital
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
        # symbols
        '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾',
        # alphabet
        'a': 'ᵃ', 'b': 'ᵇ', 'c': 'ᶜ', 'd': 'ᵈ', 'e': 'ᵉ',
        'f': 'ᶠ', 'g': 'ᵍ', 'h': 'ʰ', 'i': 'ⁱ', 'j': 'ʲ',
        'k': 'ᵏ', 'l': 'ˡ', 'm': 'ᵐ', 'n': 'ⁿ', 'o': 'ᵒ',
        'p': 'ᵖ', 'q': '۹', 'r': 'ʳ', 's': 'ˢ', 't': 'ᵗ',
        'u': 'ᵘ', 'v': 'ᵛ', 'w': 'ʷ', 'x': 'ˣ', 'y': 'ʸ',
        'z': 'ᶻ', 'A': 'ᴬ', 'B': 'ᴮ', 'D': 'ᴰ', 'E': 'ᴱ',
        'G': 'ᴳ', 'H': 'ᴴ', 'I': 'ᴵ', 'J': 'ᴶ', 'K': 'ᴷ',
        'L': 'ᴸ', 'M': 'ᴹ', 'N': 'ᴺ', 'O': 'ᴼ', 'P': 'ᴾ',
        'R': 'ᴿ', 'T': 'ᵀ', 'U': 'ᵁ', 'V': 'ⱽ', 'W': 'ᵂ'
    }

    # Unicode: subscript map
    SUBSCRIPT_MAP = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
        '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
        '+': '₊', '-': '₋', '=': '₌', '(': '₍', ')': '₎',
        'a': 'ₐ', 'e': 'ₑ', 'h': 'ₕ', 'i': 'ᵢ', 'j': 'ⱼ',
        'k': 'ₖ', 'l': 'ₗ', 'm': 'ₘ', 'n': 'ₙ', 'o': 'ₒ',
        'p': 'ₚ', 'r': 'ᵣ', 's': 'ₛ', 't': 'ₜ', 'u': 'ᵤ',
        'v': 'ᵥ', 'x': 'ₓ'
    }

    @classmethod
    def to_superscript(cls, text, format_type='simple'):
        """
        convert text to superscript

        parameters:
            text: text to convert
            format_type:
                'simple' - basic
                'chemical' - for chemical
                'math' - for math
        """
        if format_type == 'chemical':
            return cls._chemical_format(text)
        elif format_type == 'math':
            return cls._math_format(text)
        else:
            return cls._simple_convert(text, cls.SUPERSCRIPT_MAP)

    @classmethod
    def to_subscript(cls, text):
        """convert text to subscript"""
        return cls._simple_convert(text, cls.SUBSCRIPT_MAP)

    @classmethod
    def _simple_convert(cls, text, char_map):
        """basi mode"""
        result = []
        for char in text:
            if char in char_map:
                result.append(char_map[char])
            else:
                result.append(char)
        return ''.join(result)

    @classmethod
    def _chemical_format(cls, text):
        """chemical mode"""
        result = []
        i = 0
        while i < len(text):
            if i + 1 < len(text) and text[i + 1].isdigit():
                # process subscript for chemical symbols
                result.append(text[i])
                result.append(cls.to_subscript(text[i + 1]))
                i += 2
            elif text[i].isdigit():
                # process digital subscript
                result.append(cls.to_subscript(text[i]))
                i += 1
            else:
                result.append(text[i])
                i += 1
        return ''.join(result)

    @classmethod
    def _math_format(cls, text):
        """math mode"""
        # process mathematical formular
        result = []
        i = 0
        length = len(text)

        while i < length:
            char = text[i]

            # process superscript of ^
            if char == '^' and i + 1 < length:
                if text[i + 1] == '(':
                    # process format of ^(xxx)
                    end = text.find(')', i + 2)
                    if end != -1:
                        content = text[i + 2:end]
                        result.append(cls.to_superscript(content))
                        i = end + 1
                        continue
                else:
                    # process signal data
                    result.append(cls.to_superscript(text[i + 1]))
                    i += 2
                    continue

            result.append(char)
            i += 1

        return ''.join(result)

    @classmethod
    def format_expression(cls, base, exponent=None, is_subscript=False):
        """format expression：base^exponent"""
        if exponent is None:
            return base

        if is_subscript:
            return f"{base}{cls.to_subscript(exponent)}"
        else:
            return f"{base}{cls.to_superscript(exponent)}"

