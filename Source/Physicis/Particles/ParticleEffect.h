#pragma once
#include "Particle.h"
#include "AbstractEmitter.h"
#include <Scene/GameObject.h>
#include <Scene/Managers/TextureManager.h>

class Scene;

class ParticleEffect
{
public:
	ParticleEffect(Scene* scene, uint numParticles = 0);
	~ParticleEffect();

	// A vertex for the particle
	struct Vertex
	{
		Vertex()
			: m_pos(Vector3D::ZERO)
			, m_diffuse(QColor())
			, m_tex0(Vector2D::ZERO)
		{}

		vec3   m_pos;      // Vertex position
		QColor m_diffuse;  // Diffuse color
		vec2   m_tex0;     // Texture coordinate
	};

	typedef QVector<Vertex> VertexBuffer;

	void setParticleEmitter( AbstractEmitter* pEmitter );

	// Test method to randomize the particles in an interesting way
	void RandomizeParticles();
	void EmitParticles();

	virtual void update( float fDeltaTime );
	virtual void render();

	TexturePtr loadTexture( QString& fileName );
	// Resize the particle buffer with numParticles
	void resize( unsigned int numParticles );

	// Build the vertex buffer from the particle buffer
	void buildVertexBuffer();

	void linkGameObject(GameObject* go) { m_actor = go; }

protected:
	void randomizeParticle( Particle& particle );
	void emitParticle( Particle& particle );

private:
	Scene* m_scene;
	GameObject* m_actor;

	QSharedPointer<TextureManager>  m_textureManager;
	QSharedPointer<AbstractEmitter> m_particleEmitter;

	ParticleBuffer      m_particles;
	VertexBuffer        m_vertexBuffer;
	TexturePtr          m_texture;

	// Apply this force to every particle in the effect
	vec3           m_force;
};

