#include "AbstractModel.h"
AbstractModel::AbstractModel(ShadingTechniquePtr tech, const QString& fileName) 
	: Component(0),
	  m_fileName(fileName),
	  m_RenderingEffect(tech)
{
	if (tech)
	{
		m_vao = tech->getVAO();
	}
}
AbstractModel::~AbstractModel() {}