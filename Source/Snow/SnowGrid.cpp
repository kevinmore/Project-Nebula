#include "SnowGrid.h"
#include <Scene/Scene.h>

SnowGrid::SnowGrid()
	: Component(1),// get rendered last
	  m_size(0),
	  m_glIndices(0),
	  m_glVBO(0),
	  m_glVAO(0)
{
	Q_ASSERT(initializeOpenGLFunctions());
}

SnowGrid::~SnowGrid()
{
	deleteBuffers();
}

void SnowGrid::initialize()
{
	if (!m_actor)
	{
		qWarning() << "Snow Grid component initialize failed. Needs to be attached to a Game Object first.";
		return;
	}
	installShader();
}

void SnowGrid::reset()
{
	deleteBuffers();
	buildBuffers();
}

void SnowGrid::installShader()
{
	m_renderingEffect = ParticleTechniquePtr(new ParticleTechnique("snow_grid", ParticleTechnique::GENERAL));
	if (!m_renderingEffect->init()) 
	{
		qWarning() << "Snow Grid rendering effect initializing failed.";
		return;
	}
}

void SnowGrid::setGrid( const Grid &grid )
{
	m_grid = grid;
	m_size = m_grid.nodeCount();
	reset();
}

void SnowGrid::render( const float currentTime )
{
	m_renderingEffect->enable();
	m_renderingEffect->getShaderProgram()->setUniformValue("gWVP", 
		Scene::instance()->getCamera()->viewProjectionMatrix() * m_actor->getTransformMatrix());
	
// 	glDepthMask( false );
// 
// 	glEnable( GL_BLEND );
// 	glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
// 	glEnable( GL_POINT_SMOOTH );
// 
// 	glEnable( GL_ALPHA_TEST );
// 	glAlphaFunc( GL_GREATER, 0.05f );

	glBindVertexArray( m_glVAO );
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, m_glIndices );
	glDrawElements( GL_POINTS, m_size, GL_UNSIGNED_INT, (void*)(0) );
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
	glBindVertexArray( 0 );

//	glDepthMask(true);
}

bool SnowGrid::hasBuffers() const
{
	return m_glVBO > 0;
}

void SnowGrid::deleteBuffers()
{
	// Delete OpenGL VBO and unregister with CUDA
	if ( hasBuffers() ) 
	{
		glBindBuffer( GL_ARRAY_BUFFER, m_glVBO );
		glDeleteBuffers( 1, &m_glVBO );
		glDeleteVertexArrays( 1, &m_glVAO );
		glBindBuffer( GL_ARRAY_BUFFER, 0 );
		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, m_glIndices );
		glDeleteBuffers( 1, &m_glIndices );
		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
	}
	m_glVBO = 0;
	m_glVAO = 0;
	m_glIndices = 0;
}

void SnowGrid::buildBuffers()
{
	Node *data = new Node[m_size];
	memset( data, 0, m_size*sizeof(Node) );

	// Build VBO
	glGenBuffers( 1, &m_glVBO );
	glBindBuffer( GL_ARRAY_BUFFER, m_glVBO );
	glBufferData( GL_ARRAY_BUFFER, m_size*sizeof(Node), data, GL_DYNAMIC_DRAW );

	delete [] data;

	// Build VAO
	glGenVertexArrays( 1, &m_glVAO );
	glBindVertexArray( m_glVAO );

	// Mass attribute
	glEnableVertexAttribArray( 0 );
	glVertexAttribPointer( 0, 1, GL_FLOAT, GL_FALSE, sizeof(Node), (void*)(0) );

	// Velocity attribute
	glEnableVertexAttribArray( 1 );
	glVertexAttribPointer( 1, 3, GL_FLOAT, GL_FALSE, sizeof(Node), (void*)(sizeof(GLfloat)) );

	// Force attribute
	glEnableVertexAttribArray( 2 );
	glVertexAttribPointer( 2, 3, GL_FLOAT, GL_FALSE, sizeof(Node), (void*)(sizeof(GLfloat)+2*sizeof(vec3)) );

	glBindVertexArray( 0 );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );

	// Indices (needed to access vertex index in shader)
	QVector<unsigned int> indices;
	for ( unsigned int i = 0; i < (unsigned int)m_size; ++i ) indices += i;

	glGenBuffers( 1, &m_glIndices );
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, m_glIndices );
	glBufferData( GL_ELEMENT_ARRAY_BUFFER, indices.size()*sizeof(unsigned int), indices.data(), GL_DYNAMIC_DRAW );
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );

	if (m_renderingEffect)
		m_renderingEffect->setVAO(m_glVAO);
}
