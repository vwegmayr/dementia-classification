import numpy as np 
import operator
import matplotlib.pyplot as plt

class SpreadSheet(object):
    """
    Wrapper class for np.genfromtxt.
    Args:
        file: Path to csv file.
    """
    def __init__(self, file):
        super(SpreadSheet, self).__init__()

        self.sheet = np.genfromtxt(
            file,
            dtype=None,
            delimiter=",",
            comments="@@@"
            ).astype(str)

        # Contains the column names of the spreadsheet
        self.fields = list(map(lambda x: x.replace('"', ""), self.sheet[0]))

        # Contains each column separately, addressable by its field name
        self.data = {}
        for i in range(len(self.fields)):
            self.data.update({self.fields[i] : self.sheet[1:,i]})

        # Contains the unique elements for each column
        self.unique = {}
        for field in self.fields:
            u,cnts = np.unique(self.data[field], return_counts=True)
            self.unique.update({field : dict(zip(u,cnts))})

    def list_unique(self, field):
        """
        List the unique values of a column.
        Args:
            field: Name of the column header.

        Returns: Numpy array [num_unique_elements, 2]. The first column contains
        the value of each unique element, the second column contains the count
        how often it appears.
        """
        return np.array(list(reversed(
            sorted(self.unique[field].items(), key=operator.itemgetter(1))
            )))

    def plt_hist_unique(self, field):
        """
        Plot a histogram of the unique elements of a column
        Args:
            field: Name of the column header.

        Returns: None. Figure is save to current folder. By default, only unique
         values with a count larger than 100 are plotted.
        """
        Plot 
        unzipped = list(zip(*self.list_unique(field)))

        names = unzipped[0]
        cnts = unzipped[1]

        cut=0
        while cut < len(cnts) and int(cnts[cut]) > 100:
            cut+=1

        ind = range(len(cnts[:cut]))

        fig = plt.figure()

        plt.bar(ind,cnts[:cut], width=0.5)
        plt.xticks(ind, names[:cut], rotation="vertical", fontsize=10)
        plt.tight_layout()

        plt.savefig(field+".png")

    def plt_hist(self, field, **kwargs):
        """
        Plot a histogram of all values of one column. Assumed to be numeric.
        Args: 
            field: Name of the column header.

        Returns: None. Figure is save to current folder.
        """
        data = self.data[field].astype(float)

        minV=min(data)
        maxV=max(data)

        minR=np.percentile(data, q=2)
        maxR=np.percentile(data, q=98)

        fig = plt.figure()
        plt.hist(data, range=(minR,maxR), **kwargs)
        plt.title(field)
        plt.xlabel("Min=" + str(minV) + " Max=" + str(maxV))
        plt.savefig(field + ".png")