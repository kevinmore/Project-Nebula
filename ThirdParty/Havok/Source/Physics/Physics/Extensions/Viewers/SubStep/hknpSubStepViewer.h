/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SUB_STEP_VIEWER_H
#define HKNP_SUB_STEP_VIEWER_H

#include <Physics/Physics/hknpTypes.h>
#include <Physics/Physics/Extensions/Viewers/hknpViewer.h>
#include <Physics/Physics/Dynamics/Modifier/hknpModifier.h>

class hkProcessFactory;


/// Displays the intermediate positions of bodies whenever the bodies are colliding.
/// This is used for debugging live Jacobians.
class hknpSubStepViewer : public hknpViewer, public hknpModifier
{
	public:

		static inline const char* HK_CALL getName() { return HKNP_SUBSTEP_VIEWER_NAME; }

		/// Register this viewer with a factory.
		static void HK_CALL registerViewer( hkProcessFactory& factory );

		/// Create a hknpSubStepViewer.
		static hkProcess* HK_CALL create( const hkArray<hkProcessContext*>& contexts );

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		virtual ~hknpSubStepViewer();

		void addBody( hknpBodyId bodyId, hkColor::Argb color = hkColor::GRAY );
		void removeBody( hknpBodyId bodyId );
		void removeAllBodies();

		//
		// hknpViewer implementation
		//

		virtual int getProcessTag() { return s_tag; }

		virtual void init();

		virtual void step( hkReal deltaTime );

		virtual void worldAddedCallback( hknpWorld* world );

		virtual void worldRemovedCallback( hknpWorld* world );

		//
		// hknpModifier implementation
		//

		virtual int getEnabledFunctions();

		virtual void manifoldProcessCallback(
			const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
			const hknpCdBody& cdBodyA, const hknpCdBody& cdBodyB,
			hknpManifold* HK_RESTRICT manifold
			) HK_OVERRIDE;

	protected:

		hknpSubStepViewer( const hkArray<hkProcessContext*>& contexts );

	protected:

		struct BodyTransforms
		{
			enum { MAX_NUM_TRANSFORMS = 4 };
			static int m_idOffsets[MAX_NUM_TRANSFORMS];// = { 0x010000, 0x020000, 0x040000, 0x080000 };
			hkTransform m_transforms[MAX_NUM_TRANSFORMS];
			int m_nextTransform;
		};

		hkArray<hknpBodyId>			m_dynamicBodies;
		hkArray<BodyTransforms> 	m_dynamicBodyTransforms;
		hkArray<hkDisplayGeometry*> m_displayGeometries;
		hknpWorld* 					m_world;

		static int s_tag;
};


#endif // HKNP_SUB_STEP_VIEWER_H

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
