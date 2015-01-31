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

	init();

	setPolygonMode(Fill);
}
AbstractModel::~AbstractModel() {}

void AbstractModel::init()
{
	Q_ASSERT(initializeOpenGLFunctions());
}
void AbstractModel::drawElements( unsigned int index)
{

	if (m_polygonMode == Fill)
		glEnable(GL_CULL_FACE);
	else
		glDisable(GL_CULL_FACE);

	glPolygonMode(GL_FRONT_AND_BACK, m_polygonMode);

	glBindVertexArray(m_vao);

	glDrawElementsBaseVertex(
		GL_TRIANGLES,
		m_meshes[index]->getNumIndices(),
		GL_UNSIGNED_INT,
		reinterpret_cast<void*>((sizeof(unsigned int)) * m_meshes[index]->getBaseIndex()),
		m_meshes[index]->getBaseVertex()
		);

	// Make sure the VAO is not changed from the outside    
	glBindVertexArray(0);

	if (m_polygonMode == Fill)
		glDisable(GL_CULL_FACE);
	else
		glEnable(GL_CULL_FACE);
}

void AbstractModel::setPolygonMode( PolygonMode mode )
{
	m_polygonMode = mode;
}

