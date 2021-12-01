# python 命名风格：
*  `_xx`: weak "internal use" indicator. E.g. from M import * does not import objects whose name starts with an underscore.只是一种约定，该访问还可以访问
*  `xx_`:以免误使用python的关键字
*  `__xx__`: magic method 会被自动调用。
*  `__xx`: 会引起`name mangling`，以此来避免在定义子类的时候命名冲突。
    >"In compiler construction, name mangling (also called name decoration) is a technique used to solve various problems caused by the need to resolve unique names for programming entities in many modern programming languages."--wiki
    ```py
    class Person:
        def __init__(self):
            self.name = 'Sarah'
            self._age = 26
            self.__id = 30

    """
    >>> p = Person()
    >>> dir(p)
    ['_Person__id', ..., '_age', 'name']
    """

    class Employee(Person):
        def __init__(self):
            Person.__init__(self)        
            self.__id = 25
    """        
    >>> emp = Employee()
    >>> dir(emp)
    ['_Person__id', '_Employee__id', ..., '_age', 'name' ]
    """
    ```
