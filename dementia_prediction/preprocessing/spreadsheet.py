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
        how often it appears. Sorted by count.
        """
        return np.array(list(reversed(
            sorted(self.unique[field].items(), key=operator.itemgetter(1))
            )))

    def _unzip_unique(self, field):
        """Return names and counts separately"""
        unzipped = list(zip(*self.list_unique(field)))

        names = unzipped[0]
        cnts = unzipped[1]

        return names, cnts

    def plt_bar_unique(self, field):
        """
        Bar plot with counts of the unique elements of a column
        Args:
            field: Name of the column header.

        Returns: None. Figure is save to current folder. By default, only unique
         values with a count larger than 100 are plotted.
        """
        names, cnts = self._unzip_unique(field)

        cut=0
        while cut < len(cnts) and int(cnts[cut]) > 100:
            cut+=1

        ind = range(len(cnts[:cut]))

        fig = plt.figure()

        plt.bar(ind,cnts[:cut], width=0.5)
        plt.xticks(ind, names[:cut], rotation="vertical", fontsize=10)
        plt.tight_layout()

        plt.savefig(field+".png")

        plt.close(fig)

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

    def plt_bar_stack(self, field1, field2):
        """
        Bar plot with counts of the unique elements of field1, overlaid by
        counts of unique elemnts of field2.

        Helps to visualize the distribution of field2 across different unique
        elements of field1.
        Example: field1 = Gender, field2 = age. The result is a bar plot of with
        two bars (male/female) and each bar consists of the stacked counts of
        different ages.

        Args:
            field1: Name of column header.
            field2: Name of column header.

        Returns: None. Figure is save to current folder.
        """

        names1, _ = self._unzip_unique(field1)
        names2, _ = self._unzip_unique(field2)
        data1 = self.data[field1]
        data2 = self.data[field2]

        ind = range(len(names1))

        fig = plt.figure()
        plots = []
        bottom = np.zeros(len(names1))

        for name2 in names2:
            stacks = []
            for name1 in names1:
                cnt = 0
                for i in range(len(data2)):
                    if data1[i] == name1 and data2[i] == name2:
                        cnt += 1
                stacks.append(cnt)
            plot = plt.bar(ind, stacks, bottom=bottom,width=0.5)
            bottom += stacks
            plots.append(plot[0])

        plt.xticks(ind, names1, rotation="vertical", fontsize=10)
        plt.tight_layout()

        plt.legend(plots, names2)

        plt.savefig(field1+"_"+field2+"_stack"+".png")

        plt.close(fig)