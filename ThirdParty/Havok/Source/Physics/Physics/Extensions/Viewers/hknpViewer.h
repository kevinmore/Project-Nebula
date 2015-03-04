/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_VIEWER_H
#define HKNP_VIEWER_H

#include <Physics/Physics/hknpTypes.h>
#include <Common/Visualize/hkProcess.h>
#include <Physics/Physics/Extensions/Viewers/hknpProcessContext.h>

class hkProcessFactory;

#define HKNP_SHAPE_VIEWER_NAME					"NP Shapes"
#define HKNP_MANIFOLD_VIEWER_NAME				"NP Manifolds"
#define HKNP_BROAD_PHASE_VIEWER_NAME			"NP Broad Phase"
#define HKNP_CELL_VIEWER_NAME					"NP Cells"
#define HKNP_DEACTIVATION_VIEWER_NAME			"NP Deactivation"
#define HKNP_MASS_PROPERTIES_VIEWER_NAME		"NP Mass Properties"
#define HKNP_MOTION_TRAIL_VIEWER_NAME			"NP Motion Trails"
#define HKNP_CONSTRAINT_VIEWER_NAME				"NP Constraints"
#define HKNP_BODY_ID_VIEWER_NAME				"NP Body IDs"
#define HKNP_BOUNDING_RADIUS_VIEWER_NAME		"NP Bounding Radii"
#define HKNP_WELDING_TRIANGLE_VIEWER_NAME		"NP Welded Triangles"
#define HKNP_COMPOSITE_QUERY_AABB_VIEWER_NAME	"NP Composite Query AABB"
#define HKNP_MOTION_ID_VIEWER_NAME				"NP Motion IDs"
#define HKNP_SUBSTEP_VIEWER_NAME				"NP Sub Steps"


/// Base class of all Physics viewers
class hknpViewer : public hkReferencedObject, public hkProcess, protected hknpProcessContextListener
{
	public:

		static void HK_CALL displayOrientedPoint( hkDebugDisplayHandler* displayHandler, hkVector4Parameter position, const hkRotation& rot, hkSimdRealParameter size, hkColor::Argb color, int id, int tag );

		static void HK_CALL displayArrow( hkDebugDisplayHandler* displayHandler, hkVector4Parameter startPos, hkVector4Parameter arrowDirection, hkVector4Parameter perpDirection, hkColor::Argb color, hkSimdRealParameter scale, int id, int tag );

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		/// Constructor. Registers as a context listener with any Physics context(s) provided.
		hknpViewer( const hkArray<hkProcessContext*>& contexts );

		/// Destructor. Unregisters as a context listener.
		virtual ~hknpViewer();

		/// Set a viewer specific body ID to provide the option of closer analysis of individual bodies.
		void setViewerSpecificBody( const hknpWorld* world, hknpBodyId bodyId ) { m_selectedBody = bodyId; m_worldForViewerSpecificBody = world; onSetViewerSpecificBody(); }

		/// Override to reset analysis data when the viewer specific body is changed.
		virtual void onSetViewerSpecificBody() {}

		//
		// Default hkProcess implementation
		//

		/// Calls worldAddedCallback() for any worlds that are already present in the context.
		virtual void init();

		virtual void step( hkReal deltaTime ) {}

		virtual void getConsumableCommands( hkUint8*& commands, int& numCommands ) { commands = HK_NULL; numCommands = 0; }

		virtual void consumeCommand( hkUint8 command ) {}

		//
		// Default hknpProcessContextListener implementation
		//

		virtual void worldAddedCallback( hknpWorld* newWorld ) {}

		virtual void worldRemovedCallback( hknpWorld* newWorld ) {}

		virtual hkProcess* getProcess() HK_OVERRIDE { return this; }

	protected:

		hknpProcessContext* m_context;

		hknpBodyId m_selectedBody;
		const hknpWorld* m_worldForViewerSpecificBody;
};


#endif // HKNP_VIEWER_H

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
