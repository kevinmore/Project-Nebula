/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_BROAD_PHASE_CONFIG_H
#define HKNP_BROAD_PHASE_CONFIG_H

#include <Physics/Physics/hknpTypes.h>


/// An interface for configuring broad phase behavior.
/// This is used to configure a number of broad phase "layers" at broad phase construction time, and assign bodies to
/// those layers dynamically during simulation. The configuration of the layers determines which pairs can collide,
/// and whether the bodies within each layer are expected to often be moving.
/// Note: The performance of collision queries (e.g. ray casts) may suffer if using many layers. Please ensure that
/// your hknpCollisionFilter implementation filters your queries based on hknpBroadPhaseLayerIndex where possible.
class hknpBroadPhaseConfig : public hkReferencedObject
{
	public:

		/// A layer configuration.
		struct Layer
		{
			HK_DECLARE_REFLECTION();

			/// A mask of layer indices that this layer should collide with.
			hkUint32 m_collideWithLayerMask;

			/// Whether the broad phase representation of all bodies in this layer should be refreshed in every step.
			/// This should be set to true if the layer can contain dynamic bodies.
			hkBool m_isVolatile;
		};

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		/// Empty constructor.
		hknpBroadPhaseConfig() {}

		/// Serialization constructor.
		hknpBroadPhaseConfig( hkFinishLoadedObjectFlag f );

		/// Destructor.
		virtual ~hknpBroadPhaseConfig() {}

		/// Get the number of layers.
		/// This is called during broad phase construction.
		virtual int getNumLayers() const = 0;

		/// Get the configuration of a given layer index.
		/// This is called during broad phase construction.
		virtual const Layer& getLayer( hknpBroadPhaseLayerIndex index ) const = 0;

		/// Get the layer indices of a batch of bodies.
		/// This is called whenever some bodies are marked as dirty in terms of which layer they belong to.
		/// See hknpBroadPhase::markBodiesDirty().
		virtual void getLayerIndices(
			const hknpBodyId* bodyIds, int numBodyIds, hknpBody* bodies,
			hknpBroadPhaseLayerIndex* layerIndicesOut ) const = 0;
};


/// The default broad phase configuration.
/// The following layers are defined:
///  - Static: For all static bodies
///  - Dynamic: For all active dynamic bodies
///  - Inactive: For all inactive dynamic bodies
///  - Query: For all non-colliding bodies
/// The following collisions are enabled:
///  - Dynamic vs Static
///  - Dynamic vs Dynamic
///  - Dynamic vs Inactive
class hknpDefaultBroadPhaseConfig : public hknpBroadPhaseConfig
{
	public:

		/// Layer indices.
		enum
		{
			LAYER_DYNAMIC,		///< For active dynamic bodies. Should be first.
			LAYER_QUERY,		///< For bodies with the DONT_COLLIDE flag set.
			LAYER_INACTIVE,		///< For inactive dynamic bodies.
			LAYER_STATIC,		///< For static bodies.	Should be last.
			NUM_LAYERS
		};

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		/// Constructor. Sets up the layers.
		hknpDefaultBroadPhaseConfig();

		/// Serialization constructor.
		hknpDefaultBroadPhaseConfig( hkFinishLoadedObjectFlag f );

		//
		// hknpBroadPhaseConfig interface
		//

		virtual int getNumLayers() const HK_OVERRIDE;

		virtual const Layer& getLayer( hknpBroadPhaseLayerIndex index ) const HK_OVERRIDE;

		virtual void getLayerIndices(
			const hknpBodyId* bodyIds, int numBodyIds, hknpBody* bodies,
			hknpBroadPhaseLayerIndex* layerIndicesOut ) const HK_OVERRIDE;

	protected:

		Layer m_layers[NUM_LAYERS];
};


#endif // HKNP_BROAD_PHASE_CONFIG_H

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
