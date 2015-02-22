#include "ParticleEffect.h"
#include <Scene/Scene.h>

ParticleEffect::ParticleEffect(Scene* scene, uint numParticles /* = 0 */)
	: m_particleEmitter(0),
	  m_actor(0),
	  m_texture(0),
	  m_force(vec3(0.0f, -9.8f, 0.0f)),
	  m_scene(scene)
{
	m_textureManager = m_scene->textureManager();
	resize(numParticles);
}


ParticleEffect::~ParticleEffect()
{
}

void ParticleEffect::update( float fDeltaTime )
{

}

void ParticleEffect::render()
{

}

void ParticleEffect::setParticleEmitter( AbstractEmitter* pEmitter )
{

}

TexturePtr ParticleEffect::loadTexture( QString& fileName )
{
	TexturePtr pTexture = m_textureManager->getTexture(fileName);
	if(!pTexture)
	{
		pTexture = m_textureManager->addTexture(fileName, fileName);
	}
	return pTexture;
}

void ParticleEffect::randomizeParticle( Particle& particle )
{
	particle.m_age = 0.0f;
	particle.m_lifeTime = Random::random( 3, 5 );

	vec3 unitVec = Random::randUnitVec3();

	particle.m_position = unitVec;
	particle.m_velocity = unitVec * Random::random( 10, 20 );
}

void ParticleEffect::RandomizeParticles()
{
	foreach(Particle p, m_particles)
	{
		randomizeParticle(p);
	}
}

void ParticleEffect::emitParticle( Particle& particle )
{
	assert( m_particleEmitter != NULL );
	m_particleEmitter->emitParticle( particle );
}

void ParticleEffect::EmitParticles()
{
	if (m_particleEmitter)
	{
		foreach(Particle p, m_particles)
		{
			emitParticle(p);
		}
	}
	else
	{
		RandomizeParticles();
	}
}


void ParticleEffect::resize( unsigned int numParticles )
{
	m_particles.resize(numParticles);
	m_vertexBuffer.resize(numParticles);
}
