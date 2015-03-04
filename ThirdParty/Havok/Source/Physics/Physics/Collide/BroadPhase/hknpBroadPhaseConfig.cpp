/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Collide/BroadPhase/hknpBroadPhaseConfig.h>


hknpBroadPhaseConfig::hknpBroadPhaseConfig( hkFinishLoadedObjectFlag f )
	: hkReferencedObject(f)
{

}

hknpDefaultBroadPhaseConfig::hknpDefaultBroadPhaseConfig( hkFinishLoadedObjectFlag f )
	: hknpBroadPhaseConfig(f)
{

}

hknpDefaultBroadPhaseConfig::hknpDefaultBroadPhaseConfig()
{
	m_layers[LAYER_STATIC].m_collideWithLayerMask = (1<<LAYER_DYNAMIC);
	m_layers[LAYER_STATIC].m_isVolatile = false;

	m_layers[LAYER_DYNAMIC].m_collideWithLayerMask = (1<<LAYER_DYNAMIC);
	m_layers[LAYER_DYNAMIC].m_isVolatile = true;

	m_layers[LAYER_INACTIVE].m_collideWithLayerMask = (1<<LAYER_DYNAMIC);
	m_layers[LAYER_INACTIVE].m_isVolatile = false;

	m_layers[LAYER_QUERY].m_collideWithLayerMask = 0;	// don't collide with any layers (including itself)
	m_layers[LAYER_QUERY].m_isVolatile = true;
}

int hknpDefaultBroadPhaseConfig::getNumLayers() const
{
	return NUM_LAYERS;
}

const hknpBroadPhaseConfig::Layer& hknpDefaultBroadPhaseConfig::getLayer( hknpBroadPhaseLayerIndex index ) const
{
	return m_layers[index];
}

void hknpDefaultBroadPhaseConfig::getLayerIndices(
	const hknpBodyId* bodyIds, int numBodyIds, hknpBody* bodies,
	hknpBroadPhaseLayerIndex* layerIndicesOut ) const
{
	for( int i=0; i<numBodyIds; i++ )
	{
		const hknpBody& body = bodies[ bodyIds[i].value() ];

		if( body.m_flags.get( hknpBody::DONT_COLLIDE ) )
		{
			layerIndicesOut[i] = LAYER_QUERY;
		}
		else if( body.isActive() )	// active is most important, so move it close to the top
		{
			layerIndicesOut[i] = LAYER_DYNAMIC;
		}
		else if( body.isStatic() )
		{
			layerIndicesOut[i] = LAYER_STATIC;
		}
		else //if( body.isInactive() )
		{
			layerIndicesOut[i] = LAYER_INACTIVE;
		}
	}
}

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
