/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Dispatch/hkpAgentRegisterUtil.h>

#include <Physics2012/Collide/Agent/ConvexAgent/Gjk/hkpGskfAgent.h>
#include <Physics2012/Collide/Agent/ConvexAgent/Gjk/hkpPredGskfAgent.h>

#include <Physics2012/Collide/Agent/ConvexAgent/BoxBox/hkpBoxBoxAgent.h>
#include <Physics2012/Collide/Agent/MiscAgent/Bv/hkpBvAgent.h>
#include <Physics2012/Collide/Agent/CompoundAgent/BvTree/hkpBvTreeAgent.h>
#include <Physics2012/Collide/Agent/CompoundAgent/BvTreeStream/hkpMoppBvTreeStreamAgent.h>

#include <Physics2012/Collide/Agent/CompoundAgent/BvTree/hkpMoppAgent.h>
#include <Physics2012/Collide/Agent/CompoundAgent/BvTree/hkpBvCompressedMeshAgent.h>
#include <Physics2012/Collide/Agent/CompoundAgent/BvTree/hkpStaticCompoundAgent.h>
#include <Physics2012/Collide/Agent/MiscAgent/Phantom/hkpPhantomAgent.h>
#include <Physics2012/Collide/Agent/HeightFieldAgent/hkpHeightFieldAgent.h>

#include <Physics2012/Collide/Agent/ConvexAgent/SphereSphere/hkpSphereSphereAgent.h>
#include <Physics2012/Collide/Agent/ConvexAgent/SphereCapsule/hkpSphereCapsuleAgent.h>
#include <Physics2012/Collide/Agent/ConvexAgent/SphereTriangle/hkpSphereTriangleAgent.h>

#include <Physics2012/Collide/Agent/ConvexAgent/CapsuleCapsule/hkpCapsuleCapsuleAgent.h>
#include <Physics2012/Collide/Agent/ConvexAgent/CapsuleTriangle/hkpCapsuleTriangleAgent.h>
#include <Physics2012/Collide/Agent/ConvexAgent/SphereBox/hkpSphereBoxAgent.h>
#include <Physics2012/Collide/Agent/Deprecated/MultiSphereTriangle/hkpMultiSphereTriangleAgent.h>
#include <Physics2012/Collide/Agent/MiscAgent/MultirayConvex/hkpMultiRayConvexAgent.h>

#include <Physics2012/Collide/Agent/MiscAgent/Transform/hkpTransformAgent.h>
#include <Physics2012/Collide/Agent/CompoundAgent/List/hkpListAgent.h>
#include <Physics2012/Collide/Agent/Deprecated/ConvexList/hkpConvexListAgent.h>
#include <Physics2012/Collide/Agent/CompoundAgent/ShapeCollection/hkpShapeCollectionAgent.h>
#include <Physics2012/Collide/Agent/Deprecated/MultiSphere/hkpMultiSphereAgent.h>

#include <Physics2012/Collide/Agent/CompoundAgent/BvTreeStream/hkpBvTreeStreamAgent.h>
#include <Physics2012/Collide/Dispatch/Agent3Bridge/hkpAgent3Bridge.h>

#include <Physics2012/Collide/Agent3/PredGskAgent3/hkpPredGskAgent3.h>
#include <Physics2012/Collide/Agent3/PredGskCylinderAgent3/hkpPredGskCylinderAgent3.h>
#include <Physics2012/Collide/Agent3/CapsuleTriangle/hkpCapsuleTriangleAgent3.h>
#include <Physics2012/Collide/Agent3/BoxBox/hkpBoxBoxAgent3.h>
#include <Physics2012/Collide/Agent3/List3/hkpListAgent3.h>
#include <Physics2012/Collide/Agent3/ConvexList3/hkpConvexListAgent3.h>
#include <Physics2012/Collide/Agent3/BvTree3/hkpBvTreeAgent3.h>
#include <Physics2012/Collide/Agent3/CollectionCollection3/hkpCollectionCollectionAgent3.h>
#include <Physics2012/Collide/Agent3/Collection3/hkpCollectionAgent3.h>


void HK_CALL hkpAgentRegisterUtil::_registerBvTreeAgents(hkpCollisionDispatcher* dis)
{
	// hkpBvTreeAgent gets special treatment, as it overrides several
	// hkpBvAgent entries, which will cause an assert 
	dis->setEnableChecks( false );

	// Register bvTree against everything, and bvTree vs bvTree special case to create a bvTree agent for the larger tree
	hkpBvTreeAgent::registerAgent(dis);

	// Register MOPP against everything (already done), and MOPP vs MOPP special case (using size of MOPP code) to create a bvTree agent for the larger tree
	// Also replaces the linear cast static function for MOPP
	hkpMoppAgent::registerAgent(dis);

	// Register stream bvtree for bvTree against convex objects	.		
	hkpBvTreeStreamAgent::registerAgent(dis);

	// Replaces the linear cast static function for MOPP.
	hkpMoppBvTreeStreamAgent::registerAgent(dis);

	// Register StaticMesh agent to use the optimized linear cast function instead of the BvTree one and to provide SPU support
	hkpBvCompressedMeshAgent::registerAgent(dis);

	// Register StaticCompound agent to provide SPU support not included in BvTreeAgent
	hkpStaticCompoundAgent::registerAgent(dis);

	// This must come last
	hkBvTreeAgent3::registerAgent3(dis);

	dis->setEnableChecks( true );
}

void HK_CALL hkpAgentRegisterUtil::_registerListAgents( hkpCollisionDispatcher* dis)
{
	// Register list agent against everything else (overrides shape collection agent)
	hkpListAgent::registerAgent( dis );

	// Register the new optimized collection-simple shape agent
	// Checks must be disabled because we're overwriting the hkpListAgent entries.
	dis->setEnableChecks( false );
	hkpCollectionAgent3::registerAgent3(dis);
	hkpCollectionCollectionAgent3::registerAgent3(dis); 
	dis->setEnableChecks( true );

	// Register the convex list for hkConvexList shapes against convex shapes
	// This dispatches to a special dispatch function in hkpConvexListAgent for hkpConvexShape vs hkpConvexListShape
	// The convex list shape can be treated as a list, a convex list, or a convex object selected on a per
	// collision basis - see the dispatch function for details.
	hkpConvexListAgent::registerAgent( dis );
	hkConvexListAgent3::registerAgent3( dis );


	// This dispatches to a special dispatch function in bvTreeStream for hkpBvTreeShape vs hkpConvexListShape
	// The convex list shape can be treated as a list, a convex list, or a convex object selected on a per
	// collision basis - see the dispatch function for details.
	//		hkpBvTreeStreamAgent::registerConvexListAgent(dis);
	if(hkpCollisionAgent::registerHeightFieldAgent)
	{
		hkpCollisionAgent::registerHeightFieldAgent( dis );
	}

	//hkListAgent3 is deprecated
	//hkListAgent3::registerAgent3(dis);
}


void HK_CALL hkpAgentRegisterUtil::_registerTerminalAgents( hkpCollisionDispatcher* dis)
{

	//
	//	Default Convex - convex agents
	//
	{
		hkpPredGskfAgent::registerAgent(dis);
		hkPredGskAgent3::registerAgent3( dis );
	}

	//
	//	Special agents
	//
	{
		//
		//	Some old style agents which are not supported on the spu
		//  (unlikely we will port some agent3 over later (only if sony adds more memory to the spu))
		//
#if !defined(HK_PLATFORM_HAS_SPU)

		hkPredGskCylinderAgent3::registerAgent3( dis );

		// Warning: The box-box agent fail for small object (say 3cm in size).
#ifndef HK_PLATFORM_PSVITA // Still an optimizaion bug in BoxBiox agent with th 0.940 compiler etc
		hkpBoxBoxAgent::registerAgent(dis);
		hkBoxBoxAgent3::registerAgent3(dis);
#endif
		hkpSphereSphereAgent::registerAgent(dis);
		hkpSphereCapsuleAgent::registerAgent(dis);

		// As the hkpSphereTriangleAgent does not weld, we have use the hkPredGskAgent3 agent for agent3 streams
		hkpSphereTriangleAgent::registerAgent2(dis);

		hkpSphereBoxAgent::registerAgent(dis);
		hkpCapsuleCapsuleAgent::registerAgent(dis); // This agent must be disabled when serializing contact points. 

		hkpCapsuleTriangleAgent::registerAgent(dis);
		hkCapsuleTriangleAgent3::registerAgent3( dis );	// could be ported
		hkpMultiSphereTriangleAgent::registerAgent(dis);

		// deprecated 
		//hkpMultiRayConvexAgent::registerAgent(dis);
		//hkpBvTreeStreamAgent::registerMultiRayAgent(dis);
#else 

		//hkpSphereSphereAgent::registerAgent2(dis);	// there are not that many sphere-sphere situations and GSK is pretty fast with spheres, so we disable sphere-sphere for the time being

		hkpSphereTriangleAgent::registerAgent2(dis);
		hkpCapsuleTriangleAgent::registerAgent2(dis);
#endif
	}
}


// Register agents
void HK_CALL hkpAgentRegisterUtil::registerAllAgents(hkpCollisionDispatcher* dis)
{
	hkpRegisterAlternateShapeTypes(dis);

	//
	//	Warning: order of registering agents is important, later entries override earlier entries
	//


	//
	//	Unary agents handling secondary type
	//
	{
		hkpBvAgent::registerAgent(dis);
		hkpMultiSphereAgent::registerAgent( dis );
	}

	_registerBvTreeAgents(dis);

	//
	//	Our new midphase agent
	//
	{
		dis->setEnableChecks( false );		// This will override the shape collection vs bvTree (used to be set to bvTree), to be hkpShapeCollectionAgent
		hkpShapeCollectionAgent::registerAgent(dis);
		dis->setEnableChecks( true );
	}
	
	_registerListAgents( dis );

	hkpTransformAgent::registerAgent(dis);	
	hkpPhantomAgent::registerAgent(dis);

	_registerTerminalAgents( dis );

#if defined (HK_PLATFORM_HAS_SPU)
	HK_ASSERT2(0xad234123, hkpCollectionCollectionAgent3::g_agentRegistered, "The hkpCollectionCollectionAgent3 must be registered on PS3.");
#endif

	//dis->debugPrintTable();
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
