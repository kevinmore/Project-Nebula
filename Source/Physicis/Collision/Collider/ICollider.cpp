#include "ICollider.h"
#include <Utility/ModelLoader.h>
#include <Scene/Scene.h>

ICollider::ICollider( const vec3& center, Scene* scene )
	: Component(0),
	  m_center(center),
	  m_scene(scene)
{
	init();
}

void ICollider::init()
{
	Q_ASSERT(initializeOpenGLFunctions());

	m_renderingEffect = ShadingTechniquePtr(new ShadingTechnique("bounding_volume"));
	// the initial color is green, when it collides, it goes to red
	m_renderingEffect->enable();
	m_renderingEffect->setMatEmissiveColor(Qt::green);
}


void ICollider::render( const float currentTime )
{
	m_renderingEffect->enable();

	mat4 modelMatrix = m_actor->getTransformMatrix() * m_transformMatrix;
	m_renderingEffect->setEyeWorldPos(m_scene->getCamera()->position());
	m_renderingEffect->setMVPMatrix(m_scene->getCamera()->viewProjectionMatrix() * modelMatrix);
	m_renderingEffect->setModelMatrix(modelMatrix); 
	m_renderingEffect->setViewMatrix(m_scene->getCamera()->viewMatrix());

	GLint oldCullFaceMode, oldPolygonMode;
	glGetIntegerv(GL_CULL_FACE_MODE, &oldCullFaceMode);
	glGetIntegerv(GL_POLYGON_MODE, &oldPolygonMode);

	glDisable(GL_CULL_FACE);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	for(int i = 0; i < m_meshes.size(); ++i)
	{
		drawElements(i);
	}

	glCullFace(oldCullFaceMode);        
	glPolygonMode(GL_FRONT_AND_BACK, oldPolygonMode);

}

void ICollider::drawElements( uint index )
{
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
}

void ICollider::setColor(const QColor& col)
{
	m_renderingEffect->enable();
	m_renderingEffect->setMatEmissiveColor(col);
}

vec3 ICollider::getLocalSupportPoint( const vec3& dir, float margin /*= 0*/ ) const
{
	vec3 supportPoint;

	if ( m_colliderType == COLLIDER_POINT )
	{
		if ( dir.lengthSquared() > 0.0f )
		{
			vec3 dirN = dir.normalized();
			supportPoint = margin*dirN;
		}
		else
			supportPoint = vec3(margin, 0, 0);
	}
	else if ( m_colliderType == COLLIDER_LINESEGMENT || m_colliderType == COLLIDER_CONVEXHULL )
	{
		float maxDot = -FLT_MAX;

		for ( int i = 0; i < (int)m_vertices.size(); i++ )
		{
			const vec3& vertex = m_vertices[i];
			float dot = vec3::dotProduct(vertex, dir);

			if ( dot > maxDot )
			{
				supportPoint = vertex;
				maxDot = dot;
			}
		}
	}
	else if ( m_colliderType == COLLIDER_SPHERE )
	{
		float radius = m_halfExtent.x();

		if ( dir.lengthSquared() > 0.0 )
			supportPoint = (radius + margin) * dir.normalized();
		else
			supportPoint = vec3(0, radius + margin, 0);
	}
	else if ( m_colliderType == COLLIDER_BOX )
	{		
		supportPoint.setX(dir.x() < 0 ? -m_halfExtent.x() - margin : m_halfExtent.x() + margin);
		supportPoint.setY(dir.y() < 0 ? -m_halfExtent.y() - margin : m_halfExtent.y() + margin);
		supportPoint.setZ(dir.z() < 0 ? -m_halfExtent.z() - margin : m_halfExtent.z() + margin);
	}
	else if ( m_colliderType == COLLIDER_CONE )
	{
		float radius = m_halfExtent.x();
		float halfHeight = 2.0*m_halfExtent.y();
		float sinTheta = radius / (sqrt(radius * radius + 4 * halfHeight * halfHeight));
		const vec3& v = dir;
		float sinThetaTimesLengthV = sinTheta * v.length();

		if ( v.y() >= sinThetaTimesLengthV) {
			supportPoint = vec3(0.0, halfHeight, 0.0);
		}
		else {
			float projectedLength = sqrt(v.x() * v.x() + v.z() * v.z());
			if (projectedLength > 1e-10) {
				float d = radius / projectedLength;
				supportPoint = vec3(v.x() * d, -halfHeight, v.z() * d);
			}
			else {
				supportPoint = vec3(radius, -halfHeight, 0.0);
			}
		}

		// Add the margin to the support point
		if (margin != 0.0) {
			vec3 unitVec(0.0, -1.0, 0.0);
			if (v.lengthSquared() > 1e-10 * 1e-10) {
				unitVec = v.normalized();
			}
			supportPoint += unitVec * margin;
		}
	}


	return supportPoint;
}
