/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_WELDING_MODIFIER_H
#define HKNP_WELDING_MODIFIER_H

#include <Physics/Physics/Dynamics/Modifier/hknpModifier.h>


/// This welding method uses extra information stored with the triangle to archive welding.
/// The default implementation simply uses the triangle normal of the one sided triangles.
class hknpTriangleWeldingModifier : public hknpWeldingModifier
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpTriangleWeldingModifier );

		hknpTriangleWeldingModifier() {}

		virtual void postMeshCollideCallback(
			const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData, const WeldingInfo& wInfo,
			const hknpCdBody& cdBodyA, const hknpCdBodyBase& cdBodyB,
			hknpManifold* HK_RESTRICT manifold, int numManifolds ) HK_OVERRIDE;
};


/// Welds internal edges by comparing collision normals from neighboring triangles.
/// See hknpMaterial::m_weldingTolerance for related details.
class hknpNeighborWeldingModifier : public hknpWeldingModifier
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpNeighborWeldingModifier );

		hknpNeighborWeldingModifier() {}

		virtual void postMeshCollideCallback(
			const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData, const WeldingInfo& wInfo,
			const hknpCdBody& cdBodyA, const hknpCdBodyBase& cdBodyB,
			hknpManifold* HK_RESTRICT manifold, int numManifolds ) HK_OVERRIDE;
};


/// Welds internal edges by using the current velocity as a prediction for the future velocity.
/// This welds each triangle separately. Because of this the system might 'steal' time from the object.
/// See hknpMaterial::m_weldingTolerance for related details.
class hknpMotionWeldingModifier : public hknpWeldingModifier
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpMotionWeldingModifier );

		hknpMotionWeldingModifier() {}

		virtual void postMeshCollideCallback(
			const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData, const WeldingInfo& wInfo,
			const hknpCdBody& cdBodyA, const hknpCdBodyBase& cdBodyB,
			hknpManifold* HK_RESTRICT manifold, int numManifolds ) HK_OVERRIDE;
};

#endif	//!HKNP_WELDING_MODIFIER_H

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
