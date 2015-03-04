/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_VIEWER_COLOR_SCHEME_H
#define HKNP_VIEWER_COLOR_SCHEME_H

#include <Physics/Physics/hknpTypes.h>
#include <Common/Base/Types/Color/hkColor.h>

class hknpViewer;

/// An interface for coloring objects, used by some viewers.
class hknpViewerColorScheme
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_VDB, hknpViewerColorScheme );

		virtual ~hknpViewerColorScheme() {}

		/// Set the color for a body.
		virtual void setBodyColor( const hknpWorld* world, hknpBodyId bodyId, hkColor::Argb color, hknpViewer* viewer = HK_NULL ) = 0;

		/// Clear the color for a body.
		virtual void clearBodyColor( const hknpWorld* world, hknpBodyId bodyId, hknpViewer* viewer = HK_NULL ) = 0;

		/// Get the color for a body.
		virtual hkColor::Argb getBodyColor( const hknpWorld* world, hknpBodyId bodyId, hknpViewer* viewer = HK_NULL ) const = 0;

		/// Returns true if the body is visible and should be displayed
		virtual bool isBodyVisible( const hknpWorld* world, hknpBodyId bodyId, hknpViewer* viewer = HK_NULL ) const = 0;
};

/// Default viewer color scheme.
/// Returns colors based on high level body 'types', and allows for overriding the color of specific bodies.
class hknpDefaultViewerColorScheme : public hknpViewerColorScheme
{
	public:

		// Default Colors
		static const hkColor::Argb STATIC_BODY_COLOR;
		static const hkColor::Argb DYNAMIC_BODY_COLOR;
		static const hkColor::Argb KEYFRAMED_BODY_COLOR;
		static const hkColor::Argb TRIGGER_VOLUME_COLOR;
		static const hkColor::Argb DONT_COLLIDE_COLOR;

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_VDB, hknpDefaultViewerColorScheme );

		/// Constructor.
		hknpDefaultViewerColorScheme();

		//
		// hknpViewerColorScheme implementation
		//

		virtual void setBodyColor( const hknpWorld* world, hknpBodyId bodyId, hkColor::Argb color, hknpViewer* viewer = HK_NULL ) HK_OVERRIDE;

		virtual void clearBodyColor( const hknpWorld* world, hknpBodyId bodyId, hknpViewer* viewer = HK_NULL ) HK_OVERRIDE;

		virtual hkColor::Argb getBodyColor( const hknpWorld* world, hknpBodyId bodyId, hknpViewer* viewer = HK_NULL ) const HK_OVERRIDE;

		virtual bool isBodyVisible( const hknpWorld* world, hknpBodyId bodyId, hknpViewer* viewer = HK_NULL ) const HK_OVERRIDE;

	public:

		hkColor::Argb m_staticBodyColor;
		hkColor::Argb m_dynamicBodyColor;
		hkColor::Argb m_keyframedBodyColor;
		hkColor::Argb m_triggerVolumeColor;
		hkColor::Argb m_dontCollideColor;

	public:

		// Unique identifier for a body in a world
		struct Uid
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_VDB, Uid );
			Uid() { hkString::memSet( this, 0, sizeof(*this) ); }

			Uid( const hknpWorld* world, hknpBodyId bodyId ) : m_world(world), m_bodyId(bodyId) {}
			void operator = ( const Uid &uid ) { m_world = uid.m_world; m_bodyId = uid.m_bodyId; }

			const hknpWorld*	m_world;
			hknpBodyId			m_bodyId;
		};

		// Operations for the UID map
		struct UIdMapOperations
		{
			inline static unsigned hash( Uid key, unsigned modulus );
			inline static void invalidate( Uid& key );
			inline static hkBool32 isValid( Uid key );
			inline static hkBool32 equal( Uid key0, Uid key1 );
		};

	protected:

		// Overridden colors
		hkMap<Uid,hkColor::Argb,UIdMapOperations> m_overriddenColors;
};


#include <Physics/Physics/Extensions/Viewers/hknpViewerColorScheme.inl>


#endif // HKNP_VIEWER_COLOR_SCHEME_H

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
