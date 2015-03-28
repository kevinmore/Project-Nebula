#pragma once
#include <QOpenGLFunctions_4_3_Core>
#include <Primitives/Component.h>
#include <QVector>
#include <QSharedPointer>
#include <Scene/ShadingTechniques/ParticleTechnique.h>
#include "SnowParticle.h"
#include "GridNode.h"

class Scene;
class GameObject;
typedef QSharedPointer<GameObject> GameObjectPtr;

class SnowGrid : public Component, protected QOpenGLFunctions_4_3_Core
{
public:
	SnowGrid();
	~SnowGrid();

	virtual QString className() { return "SnowGrid"; }
	virtual void render(const float currentTime);

	void initialize();
	void clear() { deleteBuffers(); }
	void reset();

	void setGrid( const Grid &grid );
	Grid getGrid() const { return m_grid; }

	GLuint vbo() { if ( !hasBuffers() ) buildBuffers(); return m_glVBO; }

	inline int size() const { return m_size; }
	inline int nodeCount() const { return m_size; }

private:
	bool hasBuffers() const;
	void buildBuffers();
	void deleteBuffers();

	void installShader();

	// Grid properties
	Grid m_grid;
	int m_size;

	ParticleTechniquePtr m_renderingEffect;
	GLuint m_glIndices;
	GLuint m_glVBO;
	GLuint m_glVAO;
};

typedef QSharedPointer<SnowGrid> SnowGridPtr;