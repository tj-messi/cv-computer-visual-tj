/*----------------------------------------------------------------------------*/
/*                                                                            */
/*    Copyright (c) Innovation First 2020 All rights reserved.                */
/*    Licensed under the MIT license.                                         */
/*                                                                            */
/*    Module:     ai_robot_link.h                                             */
/*    Author:     James Pearman                                               */
/*    Created:    27 August 2020                                              */
/*                                                                            */
/*----------------------------------------------------------------------------*/

#ifndef AI_ROBOT_LINK_H_
#define AI_ROBOT_LINK_H_

/*----------------------------------------------------------------------------*/
/** @file    ai_robot_link.h
  * @brief   Header for robot to robot communication
*//*--------------------------------------------------------------------------*/

namespace ai {
  class robot_link : public vex::serial_link {
      public:
        // robot link is only wireless
        robot_link( int32_t index, const char *name, vex::linkType type );
        // we never tear down a robot_link
        ~robot_link();

        // status for received and sent packets
        int32_t    get_packets(void);
        int32_t    get_errors(void);
        int32_t    get_timeouts(void);
        int32_t    get_total(void);
        int32_t    get_tx_packets(void);
        int32_t    get_tx_errors(void);

        // this demo only send simple x, y and heading information to the partner robot
        // use these functions to get/set the data
        void set_remote_location( float x, float y, float heading, int32_t status );
        void get_local_location( float &x, float &y, float &heading, int32_t &status );
        void get_remote_location( float &x, float &y, float &heading );

        // header for packets, 6 bytes
        typedef struct __attribute__((__packed__)) _packet_header {
            uint8_t    sync[2];    // synchronizing bytes, find the start of a valid packet
            uint8_t    length;     // length of map record payload, does not include header
            uint8_t    type;       // type of packet, 
            uint16_t   crc;        // crc32 of payload, this may need to be removed to allow more payload
        } packet_header;

        // sync bytes used in the header
        enum class sync_byte {
            kSync1 = 0xA5,
            kSync2 = 0x5A
        };

        // we only use one type of packet in this demo code
        // because VEXlink bandwidth is limited to approx 512 bytes/second
        // we need to utilize it carefully, not all types of data would need to be transmitted
        // frequently so alternate packets could be created to avoid sending lots of
        // redundant information.
        #define   RL_LOCATION_PACKET    1

        // payload in a type RL_LOCATION_PACKET (only type in this demo)
        // payload is 12 bytes
        typedef struct __attribute__((__packed__)) _packet_1_payload {
            float      loc_x;  
            float      loc_y;
            float      heading;  
        } packet_1_payload;

        // full type RL_LOCATION_PACKET packet, location data
        // 18 bytes total
        typedef struct __attribute__((__packed__)) _packet_1_t {
            packet_header     header;
            packet_1_payload  payload;
        } packet_1_t;

      private:
        // states for the packet decode finite state machine
        enum class comms_state {
            kStateSyncWait1   = 0,
            kStateSyncWait2,
            kStateLength,
            kStateType,
            kStateCrc,
            kStatePayload,
            kStateGoodPacket,
            kStateBadPacket,
        };

        // storage for our transmit and receive packets
        // the same code is running on both robots
        // and they both send location to the other
        packet_1_t    packet_tx_1;
        packet_1_t    packet_rx_1;

        // rx related variables
        comms_state   state;      // state of rx decoding
        int32_t       _index;
        vex::timer    timer;
        uint32_t      packets;
        uint32_t      errors;
        uint32_t      timeouts;
        uint16_t      payload_length; 
        uint16_t      payload_type; 
        uint32_t      payload_crc;
        uint32_t      calc_crc;
        uint32_t      last_packet_time;
        uint32_t      total_data_received;

        int32_t       local_gps_status;

        // local storage for decoding
        union {
          packet_1_payload  pak_1;
          uint8_t           bytes[256];  // 256 byte max packet size
        } payload;

        bool  process( uint8_t data );

        // tx related variables
        uint32_t      tx_packets;
        uint32_t      tx_errors;

        // we protect access to the data structures using mutexes
        // we don't really need this with the VEXcode cooporative scheduler
        // but it's better to be safe
        vex::mutex    rxlock;
        vex::mutex    txlock;

        // tasks to handle transnit and receive processing
        static int    rx_task( void *arg );
        static int    tx_task( void *arg );
  };
};

#endif /* AI_ROBOT_LINK_H_ */