/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SHAPE_VIEWER_H
#define HKNP_SHAPE_VIEWER_H

#include <Physics/Physics/Extensions/Viewers/hknpViewer.h>
#include <Physics/Physics/Dynamics/World/hknpWorld.h>
#include <Physics/Physics/Collide/Shape/hknpShape.h>
#include <Physics/Physics/Collide/Shape/Composite/Compound/hknpShapeInstance.h>

class hkProcessFactory;


/// A viewer that displays the shapes of bodies in any number of worlds.
class hknpShapeViewer : public hknpViewer
{
	public:

		enum BodyType
		{
			DYNAMIC_BODY	= 1<<0,
			STATIC_BODY		= 1<<1,
			ANY_BODY		= DYNAMIC_BODY | STATIC_BODY
		};

	public:

		static inline const char* HK_CALL getName() { return HKNP_SHAPE_VIEWER_NAME; }
		static inline const char* HK_CALL getLowDetailName() { return HKNP_SHAPE_VIEWER_NAME " (low detail)"; }

		/// Register this viewer with a factory.
		static void HK_CALL registerViewer( hkProcessFactory& factory );

		/// Create a hknpShapeViewer.
		static hkProcess* HK_CALL create( const hkArray<hkProcessContext*>& contexts );

		// Create a hknpShapeViewer without convex radius expansion.
		static hkProcess* HK_CALL createWithoutRadius( const hkArray<hkProcessContext*>& contexts );

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		virtual ~hknpShapeViewer();

		/// Set the convex radius display mode. Defaults to hknpShape::CONVEX_RADIUS_DISPLAY_ROUNDED.
		HK_FORCE_INLINE void setConvexRadiusDisplayMode( hknpShape::ConvexRadiusDisplayMode radiusMode );

		/// Set whether to use instancing of display objects where possible. Defaults to TRUE.
		HK_FORCE_INLINE void setInstancingEnabled( bool isEnabled );

		/// Create and cache a display geometry for a given shape.
		/// This can be used to prevent CPU spikes when adding a body with an un-instanced shape.
		void precreateDisplayGeometryForShape( const hknpShape* shape );

		/// Add a world and all of its bodies to the viewer.
		/// Keeps track of any subsequent body additions/removals/changes.
		int addWorld( hknpWorld* world );

		/// Remove a world and all of its bodies from the viewer.
		HK_FORCE_INLINE void removeWorld( const hknpWorld* world );

		/// Add a body to the viewer.
		/// This is called automatically when a world is added, but may also be called manually.
		void addBody( hknpWorld* world, hknpBodyId bodyId );

		/// Remove a body from the viewer.
		/// This is called automatically when a world is removed, but may also be called manually.
		void removeBody( const hknpWorld* world, hknpBodyId bodyId );

		/// Set the display color of a body that has been added to the viewer.
		void setBodyColor( const hknpWorld* world, hknpBodyId bodyId, hkColor::Argb color );

		/// Refresh a body's display object.
		/// If shape instance IDs are provided, just refresh the visibility state of those instances.
		void refreshBody( const hknpWorld* world, hknpBodyId bodyId, hknpShapeInstanceId* childIds = HK_NULL, int numChildIds = 0 );

		/// Recreate display objects for all bodies.
		void refreshAllBodies( const hknpWorld* world, BodyType bodyTypes = ANY_BODY );

		/// Get the display object ID for a body.
		HK_FORCE_INLINE hkUlong composeDisplayObjectId( const hknpWorld* world, hknpBodyId bodyId ) const;

		/// Get the world and body ID from a display object ID.
		HK_FORCE_INLINE void decomposeDisplayObjectId( hkUlong id, hknpWorld*& worldOut, hknpBodyId& bodyIdOut ) const;

		//
		// Signal handlers
		//

		void onBodyAddedSignal( hknpWorld* world, hknpBodyId bodyid );
		void onBodyShapeSetSignal( hknpWorld* world, hknpBodyId bodyid );
		void onStaticBodyMovedSignal( hknpWorld* world, hknpBodyId bodyid );
		void onBodyRemovedSignal( hknpWorld* world, hknpBodyId bodyid );
		void onBodySwitchStaticDynamicSignal( hknpWorld* world, hknpBodyId bodyId, bool bodyIsNowStatic );

		//
		// hknpViewer implementation
		//

		virtual int getProcessTag() { return m_tag; }

		virtual void getConsumableCommands( hkUint8*& commands, int& numCommands );

		virtual void consumeCommand( hkUint8 command );

		virtual void step( hkReal deltaTime );

		virtual void worldAddedCallback( hknpWorld* world );

		virtual void worldRemovedCallback( hknpWorld* world );

	protected:

		hknpShapeViewer( const hkArray<hkProcessContext*>& contexts );

		// World tracking
		HK_FORCE_INLINE int getWorldIndex( const hknpWorld* world ) const;
		void removeWorld( int worldIndex );

		// Body tracking
		void addBody( int worldIndex, hknpBodyId bodyId );

		// Display handler interaction
		HK_FORCE_INLINE hkUlong composeDisplayObjectId( int worldIndex, hknpBodyId bodyId ) const;
		void createDisplayObject( int worldIndex, hknpBodyId bodyId, bool useInstancing = true );
		void destroyDisplayObject( int worldIndex, hknpBodyId bodyId );

		// Mouse picking
		HK_FORCE_INLINE hkBool pickObject( hkUint64 displayObject, hkVector4Parameter worldPosition );
		HK_FORCE_INLINE void dragObject( hkVector4Parameter newWorldSpacePoint );
		HK_FORCE_INLINE void releaseObject();

	protected:

		struct BodyIdMapOperations
		{
			inline static unsigned hash( hknpBodyId key, unsigned modulus );
			inline static void invalidate( hknpBodyId& key );
			inline static hkBool32 isValid( hknpBodyId key );
			inline static hkBool32 equal( hknpBodyId key0, hknpBodyId key1 );
		};

		/// Data associated with a tracked world.
		struct WorldData
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_VDB, WorldData );

			hknpWorld*				m_world;
			hkArray<hknpBodyId>		m_dynamicBodyIds;
			hkArray<hknpBodyId>		m_staticBodyIds;
			hkArray<hknpBodyId>		m_bodiesToAdd;	// pending additions
		};

	public:

		/// Data uniquely identifying an object to be displayed.
		struct GroupKey
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_VDB, GroupKey );
			GroupKey() { hkString::memSet( this, 0, sizeof(*this) ); }

			const hknpShape* m_shape;
			int m_hash;		
			hkColor::Argb m_color;
		};

	protected:

		/// A group of display object instances.
		struct Group
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_VDB, Group );

			hkArray<hkUlong> m_displayIds;
		};

	public:

		struct GroupMapOperations
		{
			inline static unsigned hash( GroupKey key, unsigned modulus );
			inline static void invalidate( GroupKey& key );
			inline static hkBool32 isValid( GroupKey key );
			inline static hkBool32 equal( GroupKey key0, GroupKey key1 );
		};

	protected:

		static int s_tag;
		static int s_tagNoRadius;
		int m_tag;

		// Per world data
		hkArray< WorldData* > m_worldDatas;	// sparse array

		// Geometry generation
		hknpShape::ConvexRadiusDisplayMode m_radiusDisplayMode;
		hkPointerMap< const hknpShape*, int > m_precreatedGeometryMap;
		hkArray< hkArray< hkDisplayGeometry* > > m_precreatedGeometries;

		// Instancing
		hkBool										m_instancingEnabled;
		hkMap< GroupKey, int, GroupMapOperations >	m_keyToGroupMap;
		hkMap< hkUlong, int >						m_displayIdToGroupMap;
		hkArray< Group >							m_groups;

		// Mouse picking
		hknpWorld*	m_pickedWorld;
		hknpBodyId	m_pickedBodyId;
		hkVector4	m_mouseSpringWorldPosition;
		hkVector4	m_mouseSpringLocalPosition;
};

#include <Physics/Physics/Extensions/Viewers/Shape/hknpShapeViewer.inl>


#endif // HKNP_SHAPE_VIEWER_H

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
