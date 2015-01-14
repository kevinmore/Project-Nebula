#include "AbstractModel.h"
AbstractModel::AbstractModel(const QString& fileName) 
	: m_fileName(fileName),
	  m_actor(0)
{}
AbstractModel::~AbstractModel() {}