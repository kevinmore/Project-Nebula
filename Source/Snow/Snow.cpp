#include "Snow.h"
#include <Scene/Scene.h>

Snow::Snow()
	: Component(1),// get rendered last
	  m_scene(Scene::instance()),
	  m_glVBO(0),
	  m_glVAO(0)
{
	Q_ASSERT(initializeOpenGLFunctions());
	installShader();
}


Snow::~Snow()
{
	deleteBuffers();
}

void Snow::render( const float currentTime )
{
	m_snowEffect->enable();
	m_snowEffect->getShaderProgram()->setUniformValue("gWVP", m_scene->getCamera()->viewProjectionMatrix() * m_actor->getTransformMatrix());

	GLfloat oldPointSize;

	glGetFloatv(GL_VERTEX_PROGRAM_POINT_SIZE, &oldPointSize);

	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glEnable( GL_POINT_SMOOTH );

	glHint( GL_POINT_SMOOTH_HINT, GL_NICEST );

	glBindVertexArray( m_glVAO );
	glDrawArrays( GL_POINTS, 0, m_particles.size() );
	glBindVertexArray( 0 );

	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glDisable( GL_POINT_SMOOTH );
}

void Snow::installShader()
{
	m_snowEffect = ParticleTechniquePtr(new ParticleTechnique("snow", ParticleTechnique::SNOW));
	if (!m_snowEffect->init()) 
	{
		qWarning() << "Snow rendering effect initializing failed.";
		return;
	}
}

void Snow::clear()
{
	m_particles.clear();
	deleteBuffers();
}

bool Snow::hasBuffers() const
{
	return m_glVBO > 0;
}

void Snow::deleteBuffers()
{
	// Delete OpenGL VBO and unregister with CUDA
	if ( hasBuffers() ) 
	{
		glBindBuffer( GL_ARRAY_BUFFER, m_glVBO );
		glDeleteBuffers( 1, &m_glVBO );
		glDeleteVertexArrays( 1, &m_glVAO );
		glBindBuffer( GL_ARRAY_BUFFER, 0 );
	}
	m_glVBO = 0;
	m_glVAO = 0;
}

void Snow::buildBuffers()
{
	deleteBuffers();

	// Build OpenGL VAO
	glGenVertexArrays( 1, &m_glVAO );
	glBindVertexArray( m_glVAO );

	// Build OpenGL VBO
	glGenBuffers( 1, &m_glVBO );
	glBindBuffer( GL_ARRAY_BUFFER, m_glVBO );
	glBufferData( GL_ARRAY_BUFFER, m_particles.size()*sizeof(SnowParticle), m_particles.data(), GL_DYNAMIC_DRAW );

	std::size_t offset = 0; // offset within particle struct

	// Position attribute
	glEnableVertexAttribArray( 0 );
	glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, sizeof(SnowParticle), (void*)offset );
	offset += sizeof(CUDAVec3);

	// Velocity attribute
	glEnableVertexAttribArray( 1 );
	glVertexAttribPointer( 1, 3, GL_FLOAT, GL_FALSE, sizeof(SnowParticle), (void*)offset );
	offset += sizeof(CUDAVec3);

	// Mass attribute
	glEnableVertexAttribArray( 2 );
	glVertexAttribPointer( 2, 1, GL_FLOAT, GL_FALSE, sizeof(SnowParticle), (void*)offset );
	offset += sizeof(GLfloat);

	// Volume attribute
	glEnableVertexAttribArray( 3 );
	glVertexAttribPointer( 3, 1, GL_FLOAT, GL_FALSE, sizeof(SnowParticle), (void*)offset );
	offset += sizeof(GLfloat);
	offset += 2*sizeof(CUDAMat3);

	// lambda (stiffness) attribute
	offset += 2*sizeof(GLfloat); // skip to material.xi
	glEnableVertexAttribArray(4);
	glVertexAttribPointer( 4, 1, GL_FLOAT, GL_FALSE, sizeof(SnowParticle), (void*)offset);
	offset += sizeof(GLfloat);

	glBindVertexArray( 0 );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );

	m_snowEffect->setVAO(m_glVAO);
}
