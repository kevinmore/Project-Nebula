#include "AbstractModel.h"
AbstractModel::AbstractModel(const QString& fileName) 
	: Component(true, 0),
	  m_fileName(fileName)
{}
AbstractModel::~AbstractModel() {}