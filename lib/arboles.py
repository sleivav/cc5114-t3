# para crear copias de los arboles
from copy import deepcopy
# random
import random

# esta funcion dice si el argumento es una funcion o no
def is_function(f):
    return hasattr(f, "__call__")

# esto nos permite recorrer una lista de a pedazos
# ejemplo el input es [1,2,3,4,5,6,7,8]
# con n=2 esta funcion nos hara tener
# [(1,2), (3,4), (5,6), (7,8)]
def chunks(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

class Node:
    def __init__(self, function):
        # nos aseguramos que el nodo recibe una funcion
        assert is_function(function)
        self.operation = function
        # esto nos permite contar cuantos argumentos recibe la funcion
        self.num_arguments = function.__code__.co_argcount
        self.arguments = []
        
    # funcion para evaluar un nodo (calcular el resultado)
    def eval(self):
        # es importante chequear que los argumentos que nos dieron
        # coincidan con los argumentos que necesitamos
        assert len(self.arguments) == self.num_arguments
        # evaluamos los argumentos, y luego los pasamos como argumentos a
        # la operacion
        # NOTA: esto -> *[...] significa que separa los elementos de la lista
        # ejemplo si tenemos `lista = [1,2,3]`
        # print(lista) -> [1,2,3]
        # print(*lista) -> 1 2 3
        # esto se llama `unpacking`.
        # lo necesitamos porque nuestra funcion recibe N argumentos
        # no una lista de tamaño N.
        return self.operation(*[node.eval() for node in self.arguments])
    
    # hace una lista con todos los hijos
    def serialize(self):
        l = [self]
        for node in self.arguments:
            l.extend(node.serialize())
        return l
    
    # copia al nodo
    def copy(self):
        return deepcopy(self)
    
    # reemplaza el nodo por otro
    def replace(self, otherNode):
        # aqui estamos haciendo algo medio hacky.
        # la forma correcta seria tener una referencia al nodo padre y
        # reemplazarse a si mismo por otro.
        # por temas de mantener el codigo corto lo hacemos asi
        # pero no lo hagan en casa!
        assert isinstance(otherNode, Node)
        self.__class__ = otherNode.__class__
        self.__dict__ = otherNode.__dict__


# esta clase representa todos los nodos quetienen 2 argumentos
class BinaryNode(Node):
    num_args = 2
    def __init__(self, function, left, right):
        # revisamos que todo sea un nodo y agregamos a las lista de
        # argumentos
        assert isinstance(left, Node)
        assert isinstance(right, Node)
        super(BinaryNode, self).__init__(function)
        self.arguments.append(left)
        self.arguments.append(right)
        

class AddNode(BinaryNode):
    def __init__(self, left, right):
        def _add(x,y):
            return x + y
        super(AddNode, self).__init__(_add, left, right)
        
    # esta es la funcion que define como se mostrara el nodo
    # como es un nodo que REPResenta la suma, lo mostramos como suma
    def __repr__(self):
        return "({} + {})".format(*self.arguments)
        
    
class SubNode(BinaryNode):
    def __init__(self, left, right):
        def _sub(x,y):
            return x - y
        super(SubNode, self).__init__(_sub, left, right)
        
    def __repr__(self):
        return "({} - {})".format(*self.arguments)
    
    
class MaxNode(BinaryNode):
    def __init__(self, left, right):
        def _max(x,y):
            return max(x,y)
        super(MaxNode, self).__init__(_max, left, right)
        
    def __repr__(self):
        return "max({{{}, {}}})".format(*self.arguments)


class MultNode(BinaryNode):
    def __init__(self, left, right):
        def _mult(x,y):
            return x * y
        super(MultNode, self).__init__(_mult, left, right)
        
    def __repr__(self):
        return "({} * {})".format(*self.arguments)
    
    
class TerminalNode(Node):
    # Este nodo representa una hoja de arbol. Es el nodo terminal
    # por lo que no tiene argumentos
    num_args = 0
    def __init__(self, value):
        # igual tenemos que representarlo como una funcion, por como
        # diseñamos el programa. Pero aqui va a ser una funcion vacia
        def _nothind(): pass
        super(TerminalNode, self).__init__(_nothind)
        self.value = value
        
    def __repr__(self):
        return str(self.value)
    
    def eval(self):
        # la evaluacion de un nodo terminal es el valor que contiene
        return self.value


# un AST es un arbol que representa un programa, la idea aqui es tener
# un generador, que pueda generar ASTs aleatorios
class AST:
    def __init__(self, allowed_functions, allowed_terminals, prob_terminal=0.3):
        # las funciones (nodos en nuestro caso) que nuestro programa puede tener
        self.functions = allowed_functions
        # los terminales admitidos en nuestro programa. Numeros por ejemplo
        self.terminals = allowed_terminals
        # para no tener un arbol infinitamente profundo, existe una posibilidad
        # de que, a pesar de que ahora toque hacer otro sub arbol, que se ignore
        # eso y se ponga un terminal en su lugar.
        self.prob = prob_terminal

    # esta funcion ya la hemos visto, nos permite llamar al AST como si fuera
    # una funcion. max_depth es la produndidad que queremos tenga el arbol
    def __call__(self, max_depth=10):
        # aqui tenemos una funcion auxiliar. Nos permitira hacer esto recursivo
        def create_rec_tree(depth):
            # si `depth` es mayor a 0, nos toca crear un sub-arbol
            if depth > 0:
                # elegimos una funcion aleatoriamente
                node_cls = random.choice(self.functions)
                # aqui iremos dejando los argumentos que necesita la funcion
                arguments = []
                # para cada argumento que la funcion necesite...
                for _ in range(node_cls.num_args):
                    # existe un `prob` probabilidad de que no sigamos creando
                    # sub-arboles y lleguemos y cortemos aqui para hacer
                    # un nodo terminal
                    if random.random() < self.prob:
                        arguments.append(create_rec_tree(0))
                    else:
                        # la otra es seguir creando sub-arboles recursivamente
                        arguments.append(create_rec_tree(depth - 1))

                # `arguments` es una lista y los nodos necesitan argumentos
                # asi que hacemos "unpacking" de la lista
                return node_cls(*arguments)
            else:
                # si `depth` es 0 entonces creamos un nodo terminal con
                # alguno de los terminales permitidos que definimos inicialmente
                return TerminalNode(random.choice(self.terminals))

        # llamamos a la funcion auxiliar para crear un arbol de profundidad `max_depth`
        return create_rec_tree(max_depth)