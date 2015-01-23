#include "AbstractModel.h"
AbstractModel::AbstractModel(const QString& fileName) 
	: Component(0),
	  m_fileName(fileName)
{}
AbstractModel::~AbstractModel() {}