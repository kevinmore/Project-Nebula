#include "AbstractModel.h"
AbstractModel::AbstractModel(const QString& fileName) 
	: Component(),
	  m_fileName(fileName)
{}
AbstractModel::~AbstractModel() {}