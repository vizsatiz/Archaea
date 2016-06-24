from pybrain.tools.xml.networkwriter import NetworkWriter


class PyBrainNetworkPersistenceHelper:

    def __init__(self):
        pass

    @staticmethod
    def get_model_state(model_object):
        """
        Method that gets the current state of the network

        :param model_object:
        :return:
        """
        return NetworkWriterHelper.get_network_state(model_object, 'filename')

    @staticmethod
    def initialize_model_with_state(dom):
        """
        Method re-initiates the network from dom

        :param dom:
        :return:
        """
        return NetworkReaderHelper(dom).read_network_state()


class NetworkWriterHelper(NetworkWriter):
    @staticmethod
    def get_network_state(net, filename):
        """
        Returns the Network State as XML

        :param net:
        :param filename:
        :return:
        """
        w = NetworkWriter(filename, newfile=True)
        netroot = w.newRootNode('Network')
        w.writeNetwork(net, netroot)
        return w.dom


class NetworkReaderHelper:

    mothers = {}
    modules = {}

    def __init__(self, state):
        self.dom = state
        self.root = self.dom.documentElement

    def read_network_state(self, network_name=None, index=0):
        if self.dom.firstChild.nodeName != 'PyBrain':
            raise Exception, 'Not a correct PyBrain XML file'
        if network_name:
            netroot = self.findNamedNode('Network', network_name)
        else:
            netroot = self.findNode('Network', index)
        return self.readNetwork(netroot)

    def findNamedNode(self, name, nameattr, root=None):
        """ return the toplevel node with the provided name, and the fitting 'name' attribute. """
        if root == None:
            root = self.root
        for n in root.childNodes:
            if n.nodeName == name:
                if 'name' in n.attributes:
                    if n.attributes['name'] == nameattr:
                        return n
        return None

    def readModule(self, mnode):
        if mnode.nodeName == 'Network':
            m = self.readNetwork(mnode)
        else:
            m = self.readBuildable(mnode)
        self.modules[m.name] = m
        inmodule = mnode.hasAttribute('inmodule')
        outmodule = mnode.hasAttribute('outmodule')
        return m, inmodule, outmodule

    def readNetwork(self, node):
        # TODO: why is this necessary?
        import pybrain.structure.networks.custom  # @Reimport @UnusedImport
        nclass = eval(str(node.getAttribute('class')))
        argdict = self.readArgs(node)
        n = nclass(**argdict)
        n.name = node.getAttribute('name')

        for mnode in self.getChildrenOf(self.getChild(node, 'Modules')):
            m, inmodule, outmodule = self.readModule(mnode)
            if inmodule:
                n.addInputModule(m)
            elif outmodule:
                n.addOutputModule(m)
            else:
                n.addModule(m)

        mconns = self.getChild(node, 'MotherConnections')
        if mconns:
            for mcnode in self.getChildrenOf(mconns):
                m = self.readBuildable(mcnode)
                self.mothers[m.name] = m

        for cnode in self.getChildrenOf(self.getChild(node, 'Connections')):
            c, recurrent = self.readConnection(cnode)
            if recurrent:
                n.addRecurrentConnection(c)
            else:
                n.addConnection(c)

        n.sortModules()
        return n

    def findNode(self, name, index=0, root=None):
        """ return the toplevel node with the provided name (if there are more, choose the
        index corresponding one). """
        if root == None:
            root = self.root
        for n in root.childNodes:
            if n.nodeName == name:
                if index == 0:
                    return n
                index -= 1
        return None

    def readArgs(self, node):
        res = {}
        for c in self.getChildrenOf(node):
            val = c.getAttribute('val')
            if val in self.modules:
                res[str(c.nodeName)] = self.modules[val]
            elif val in self.mothers:
                res[str(c.nodeName)] = self.mothers[val]
            elif val != '':
                res[str(c.nodeName)] = eval(val)
        return res

    def getChild(self, node, name):
        """ get the child with the given name """
        for n in node.childNodes:
            if name and n.nodeName == name:
                return n

    def getChildrenOf(self, node):
        """ get the element children """
        return filter(lambda x: x.nodeType == x.ELEMENT_NODE, node.childNodes)

    def readBuildable(self, node):
        mclass = node.getAttribute('class')
        argdict = self.readArgs(node)
        try:
            m = eval(mclass)(**argdict)
        except:
            print 'Could not construct', mclass
            print 'with arguments:', argdict
            return None
        m.name = node.getAttribute('name')
        self.readParams(node, m)
        return m

    def readConnection(self, cnode):
        c = self.readBuildable(cnode)
        recurrent = cnode.hasAttribute('recurrent')
        return c, recurrent

    def readParams(self, node, m):
        import string
        pnode = self.getChild(node, 'Parameters')
        if pnode:
            params = eval(string.strip(pnode.firstChild.data))
            m._setParameters(params)
