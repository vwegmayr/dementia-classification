"""Illustrative examples for Sphinx usage"""


class DocStyles(object):
    """Summary line. Each class method implements a different style.

    The Sphinx extension napoleon converts numpy and google style docstrings
    to the sphinx format. This is why all the methods' documentation looks the
    same. But when you go view the source you will notice the difference.

    Attributes:
        attr1 (bool): Description of attr1.
        attr2 (:obj:`int`, optional): Description of attr2.

    References:
        http://www.sphinx-doc.org/en/stable/ext/napoleon.html
        #module-sphinx.ext.napoleon
    """

    def __init__(self, param1, param2):
        """Example of docstring on the __init__ method.

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1 (bool): Description of `param1`.
            param2 (:obj:`int`, optional): Description of `param2`. Multiple
                lines are supported.
        """
        self.attr1 = param1
        self.attr2 = param2

    def sphinx_style(self, arg1, arg2):
        """Summary line.

        Extended description of function.

        :param str arg1: Description of arg1.
        :param int arg2: Description of arg2.

        :return: Description of return value.
        :rtype: bool

        :raises ValueError: if arg2 exceeds 10
        :raises TypeError: if arg1 is not str
        """
        return arg1 == int(arg2) or self.attr1

    def numpy_style(self, arg1, arg2):
        """Summary line.

        Extended description of function.

        Parameters
        ----------
        arg1 : int
            Description of arg1
        arg2 : str
            Description of arg2

        Returns
        -------
        bool
            Description of return value

        Raises
        ------
        ValueError
            If arg1 exceeds 10
        TypeError
            If arg2 is not str
        """
        return arg1 == int(arg2) or self.attr1

    def google_style(self, arg1, arg2):
        """Summary line.

        Extended description of function.

        Args:
            arg1 (int): Description of arg1
            arg2 (str): Description of arg2

        Returns:
            bool: Description of return value

        Raises:
            ValueError: If arg1 exceeds 10
            TypeError: If arg2 is not str
        """
        return arg1 == int(arg2) or self.attr1
