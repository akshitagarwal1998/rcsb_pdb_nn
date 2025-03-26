from abc import ABC, abstractmethod
import numpy as np

class DescriptorInterface(ABC):
    def __init__(self, values):
        self._values = np.array(values)

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, val):
        val = np.array(val)
        if val.ndim != 1:
            raise ValueError("Descriptor must be a 1D array.")
        self._values = val

    @property
    @abstractmethod
    def descriptor_type(self):
        pass

    @abstractmethod
    def distance(self, other):
        pass


class GeometricFeature(DescriptorInterface):
    @property
    def descriptor_type(self):
        return "geometric"

    def distance(self, other):
        if not isinstance(other, GeometricFeature):
            raise TypeError("Distance comparison must be with another GeometricFeature.")
        norm_self = np.linalg.norm(self.values)
        norm_other = np.linalg.norm(other.values)
        diff_norm = np.linalg.norm(self.values - other.values)
        return (2 * diff_norm) / (1 + norm_self + norm_other)


class ZernikeMoment(DescriptorInterface):
    @property
    def descriptor_type(self):
        return "zernike"

    def distance(self, other):
        if not isinstance(other, ZernikeMoment):
            raise TypeError("Distance comparison must be with another ZernikeMoment.")
        return np.linalg.norm(self.values - other.values)


class BioZernikeMoment:
    def __init__(self, geom_values, zernike_values):
        self.geom = GeometricFeature(geom_values)
        self.zernike = ZernikeMoment(zernike_values)

    def distance_vector(self, other):
        if not isinstance(other, BioZernikeMoment):
            raise TypeError("Distance must be computed between two BioZernikeMoment objects.")
        return np.array([
            self.geom.distance(other.geom),
            self.zernike.distance(other.zernike)
        ])

    @property
    def descriptor_type(self):
        return "bioZernike"
