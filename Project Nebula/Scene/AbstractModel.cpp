#include "AbstractModel.h"
AbstractModel::AbstractModel(const QString& fileName, GameObject* go) 
	: m_fileName(fileName),
	  m_actor(go)
{}
AbstractModel::~AbstractModel() {}