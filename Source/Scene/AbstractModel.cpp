#include "AbstractModel.h"
AbstractModel::AbstractModel(ShadingTechniquePtr tech, const QString& fileName) 
	: Component(0),
	  m_fileName(fileName),
	  m_renderingEffect(tech)
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

	GLint oldCullFaceMode, oldPolygonMode;
	glGetIntegerv(GL_CULL_FACE_MODE, &oldCullFaceMode);
	glGetIntegerv(GL_POLYGON_MODE, &oldPolygonMode);

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

	glCullFace(oldCullFaceMode);        
	glPolygonMode(GL_FRONT_AND_BACK, oldPolygonMode);
}

void AbstractModel::setPolygonMode( PolygonMode mode )
{
	m_polygonMode = mode;
}

void AbstractModel::setBoundingBox( const BoxCollider& box )
{
	m_boundingBox = BoxColliderPtr(new BoxCollider(box.getCenter(), box.getGeometryShape().getHalfExtents(), box.getScene()));
}

BoxColliderPtr AbstractModel::getBoundingBox() const
{
	return m_boundingBox;
}

void AbstractModel::showBoundingBox()
{
	gameObject()->attachComponent(m_boundingBox);
}

void AbstractModel::hideBoundingBox()
{
	gameObject()->detachComponent(m_boundingBox);
}
